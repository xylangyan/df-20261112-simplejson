"""Incremental decoder for streams of concatenated JSON values.

A device that emits several JSON records back-to-back (optionally with
whitespace between them) often delivers them in arbitrary byte or text
chunks.  Framing each chunk as a record on its own is unsafe: a backslash
or a multi-byte UTF-8 sequence that straddles a boundary makes the next
chunk look like a new record.

:class:`JSONStreamDecoder` accepts those chunks explicitly via
:meth:`feed` and marks the end of the input with :meth:`eof`.  Complete
root values are returned one at a time; input that only forms a partial
value is retained until more data arrives.  Errors are raised with an
offset measured against the entire input ever fed to the decoder, not
against the individual chunk that exposed them.
"""
from __future__ import absolute_import

import codecs

from .compat import binary_type, text_type
from .decoder import JSONDecoder, WHITESPACE
from .errors import JSONDecodeError
from .scanner import NUMBER_RE

__all__ = ['JSONStreamDecoder']

# Messages that can describe a value whose remaining characters have
# simply not arrived yet.  Every other failure is an invalid document
# regardless of what the next chunk contains.
_REPAIRABLE_MESSAGES = frozenset((
    "Expecting value",
    "Expecting value or ']'",
    "Expecting property name enclosed in double quotes",
    "Expecting property name enclosed in double quotes or '}'",
    "Expecting ':' delimiter",
    "Expecting ',' delimiter or '}'",
    "Expecting ',' delimiter or ']'",
    "Unterminated string starting at",
    "Invalid \\uXXXX escape sequence",
))

# First characters of the literals (true/false/null/NaN/Infinity and the
# signed numbers) whose keyword or sign may be split across a boundary.
_LITERAL_FIRST = frozenset(u'tfnNI-')

_HEX_DIGITS = frozenset(u'0123456789abcdefABCDEF')

_NUMBER_CHARS = frozenset(u'0123456789.eE+-')


class JSONStreamDecoder(object):
    """Incrementally decode a stream of concatenated JSON documents.

    Chunks are handed to :meth:`feed` in transmission order.  Every call
    returns the (possibly empty) list of complete root values that became
    available; partial values stay buffered.  After the final chunk,
    :meth:`eof` flushes any remaining data, raising
    :class:`~simplejson.JSONDecodeError` for an incomplete value, an
    invalid value, or trailing junk.  Error positions are offsets into the
    complete input fed since construction; when an error is raised, the
    values already decoded from the same chunk are attached to the
    exception as its ``values`` attribute.

    The decoder reuses :class:`~simplejson.JSONDecoder`, so all of its
    options (``parse_float``, ``parse_int``, ``parse_constant``,
    ``object_hook``, ``object_pairs_hook``, ``array_hook``, ``strict``,
    ``allow_nan`` and ``encoding``) keep working unchanged.
    """

    def __init__(self, encoding=None, cls=None, **kw):
        """Create a stream decoder.

        *cls* defaults to :class:`~simplejson.JSONDecoder`; any subclass of
        it can be supplied.  Remaining keyword arguments are forwarded to
        that class.  A pre-configured decoder can be supplied with the
        *decoder* keyword instead.  *encoding* controls how :class:`bytes`
        chunks are interpreted (``'utf-8'`` by default).
        """
        decoder = kw.pop('decoder', None)
        if decoder is not None:
            if cls is not None or kw:
                raise TypeError(
                    "decoder cannot be combined with cls or other "
                    "decoder keyword arguments")
            if not isinstance(decoder, JSONDecoder):
                raise TypeError("decoder must be a JSONDecoder instance")
            self._decoder = decoder
        else:
            if cls is None:
                cls = JSONDecoder
            self._decoder = cls(encoding=encoding, **kw)
        self._encoding = self._decoder.encoding
        self._byte_decoder = codecs.getincrementaldecoder(self._encoding)()
        self._buffer = u''
        # Code-point offset of ``_buffer[0]`` within the whole input.
        self._base = 0
        self._closed = False

    @property
    def decoder(self):
        """The underlying :class:`~simplejson.JSONDecoder`."""
        return self._decoder

    @property
    def offset(self):
        """Code-point offset of the first unconsumed character.

        Absolute position within the concatenation of every chunk fed so
        far.
        """
        return self._base

    @property
    def closed(self):
        """Whether :meth:`eof` has been called."""
        return self._closed

    def feed(self, chunk):
        """Append *chunk* and return every complete value it completes.

        *chunk* may be :class:`str` or, when decoding bytes, :class:`bytes`.
        Bytes are decoded with the decoder's encoding, so multi-byte UTF-8
        sequences split across chunks are reassembled before parsing.
        """
        if self._closed:
            raise ValueError("feed() called on a closed stream decoder")
        text = self._to_text(chunk)
        if text:
            self._buffer += text
        return self._drain(False)

    def decode(self, chunk, final=False):
        """Feed *chunk* and, when *final* is true, mark the end of input.

        Returns the list of complete values, exactly like :meth:`feed` plus
        :meth:`eof`.  This mirrors the ``decode``/``final`` interface of
        :class:`codecs.IncrementalDecoder`; a missing chunk is treated as
        an empty chunk.
        """
        values = self.feed(u'' if chunk is None else chunk)
        if final:
            values.extend(self.eof())
        return values

    def eof(self, chunk=None):
        """Mark the input complete and return the remaining values.

        A final *chunk* behaves like a trailing :meth:`feed`.  Any buffered
        input that does not form a complete value, or that is not valid
        JSON, raises :class:`~simplejson.JSONDecodeError` with a position
        relative to the whole input.  A stream whose only unconsumed input
        is whitespace finishes cleanly.
        """
        if self._closed:
            raise ValueError("eof() called on a closed stream decoder")
        if chunk is not None:
            text = self._to_text(chunk)
            if text:
                self._buffer += text
        tail = self._finish_bytes()
        if tail:
            self._buffer += tail
        self._closed = True
        return self._drain(True)

    #: Aliases kept for readability at call sites.
    close = eof
    finish = eof
    feed_chunk = feed

    def _to_text(self, chunk):
        if isinstance(chunk, text_type):
            return chunk
        if isinstance(chunk, binary_type):
            # An incremental decoder holds back a trailing, incomplete
            # multi-byte UTF-8 sequence until the rest of it arrives.
            return self._byte_decoder.decode(chunk)
        raise TypeError(
            "chunk must be text or bytes, not %s" % type(chunk).__name__)

    def _finish_bytes(self):
        return self._byte_decoder.decode(b'', True)

    def _drain(self, at_eof):
        results = []
        decoder = self._decoder
        buffer = self._buffer
        base = self._base
        pos = 0
        while True:
            idx = WHITESPACE.match(buffer, pos).end()
            if idx == len(buffer):
                if at_eof:
                    # Nothing but whitespace is left; it is consumed too.
                    pos = idx
                break
            try:
                value, end = decoder.raw_decode(buffer, idx)
            except JSONDecodeError as err:
                if not at_eof and (self._is_incomplete(buffer, err) or
                                   self._number_is_pending(buffer, idx, err)):
                    break
                # Rebase against the buffer the scanner saw, before the
                # already-decoded prefix below shifts the buffer start, then
                # commit that prefix so it survives the raised exception.
                error = self._rebase_error(err, buffer, base)
                self._commit(buffer, base, pos)
                error.values = results
                raise error
            if not at_eof and self._number_can_continue(buffer, idx, end):
                # A number that reaches the end of the buffer, or that is
                # immediately followed by ``.``/``e``, may grow in the next
                # chunk (``2`` + ``.5`` is one number, not two records), so
                # hold it back rather than emit a truncated value.  A
                # number followed by whitespace is definitely complete.
                if end == len(buffer) or buffer[end] not in u' \t\n\r':
                    break
            results.append(value)
            pos = end
        self._commit(buffer, base, pos)
        return results

    def _commit(self, buffer, base, pos):
        if pos:
            self._buffer = buffer[pos:]
            self._base = base + pos
        else:
            self._buffer = buffer
            self._base = base

    @staticmethod
    def _number_can_continue(buffer, idx, end):
        """Whether the number ending at *end* could grow with the next chunk.

        The scanner's number regex happily accepts a prefix, so ``2``
        already looks complete when a following ``.5`` has not arrived yet.
        A number is held back when it reaches the end of the buffer and the
        next character could extend it (a decimal point or exponent after an
        integer, an exponent sign, or digits after that sign), or when a
        dangling ``.``/``e`` already follows it.  A complete float
        (``1.0``, ``1e5``) followed by a non-number character cannot grow.
        """
        match = NUMBER_RE.match(buffer, idx)
        if match is None or match.end() != end:
            return False
        if end < len(buffer):
            # The scanner stopped before the buffer ended.  An integer may
            # gain ``.`` or ``e``; a fractional number may gain an exponent.
            # Any number that already has an exponent is complete as-is.
            if match.group(3) is not None:
                return False
            return buffer[end] in u'.eE'
        # A number that reaches the end of the buffer can always be extended
        # by more digits (``12`` + ``3`` is 123, ``1e5`` + ``6`` is 1e56),
        # so it cannot be emitted until a delimiter or EOF arrives.
        return True

    @staticmethod
    def _number_is_pending(buffer, idx, err):
        """Whether a structural error is really a number cut mid-way.

        Inside a container, a dangling ``.`` or ``e`` (``[4E``,
        ``[1, 2.``) makes the scanner complain about a missing delimiter
        before it gets a chance to read the rest of the number.  The
        numeric run immediately before the error position must consist of
        number characters only and end on one that could still extend the
        same number.
        """
        if err.pos > len(buffer):
            return False
        start = err.pos
        while start > idx and buffer[start - 1] in _NUMBER_CHARS:
            start -= 1
        fragment = buffer[start:err.pos]
        if not fragment or any(char not in _NUMBER_CHARS
                               for char in fragment):
            return False
        if fragment.endswith(u'.'):
            return True
        marker = max(fragment.rfind(u'e'), fragment.rfind(u'E'))
        if marker < 0:
            # An integer is pending only while a decimal point or exponent
            # marker sits directly behind it; otherwise the scanner has
            # already finished the number and this is a genuine delimiter.
            return err.pos < len(buffer) and buffer[err.pos] in u'.eE'
        exponent = fragment[marker + 1:]
        # ``e``, ``e+`` and ``e-`` still need their exponent digits; a
        # complete exponent followed by another marker is not repairable.
        return not exponent or not exponent[-1:].isdigit()

    @staticmethod
    def _is_incomplete(buffer, err):
        msg = err.msg
        if msg not in _REPAIRABLE_MESSAGES:
            return False
        if msg == "Unterminated string starting at":
            return True
        if msg == "Invalid \\uXXXX escape sequence":
            # Repairable only while every buffered hex digit is valid and
            # fewer than four have arrived; a bad digit, or a fifth
            # character after four good ones, can never be repaired.
            pos = err.pos
            digits = buffer[pos + 2:pos + 6]
            if len(digits) < 4:
                return all(char in _HEX_DIGITS for char in digits)
            return False
        if msg == "Expecting value" or msg == "Expecting value or ']'":
            # A literal keyword or signed number may simply be cut short.
            # Any other character already buffered is a hard error.
            pos = err.pos
            return pos >= len(buffer) or buffer[pos] in _LITERAL_FIRST
        # Other structural expectations (delimiter, closing bracket,
        # property name) are repairable only while the offending position
        # is past the buffered text; a character that is already there can
        # never be replaced by a later chunk.
        return err.pos >= len(buffer)

    def _rebase_error(self, err, buffer, base):
        pos = err.pos + base
        end = None if err.end is None else err.end + base
        # Keep a prefix of the same length as the consumed input so the
        # line/column math in JSONDecodeError sees absolute positions.
        doc = (u' ' * base) + buffer
        return JSONDecodeError(err.msg, doc, pos, end)
