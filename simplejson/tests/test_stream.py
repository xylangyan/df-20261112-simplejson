# -*- coding: utf-8 -*-
from __future__ import absolute_import

import random
from decimal import Decimal
from unittest import TestCase

import simplejson as json
from simplejson import JSONDecodeError, JSONStreamDecoder
from simplejson.compat import b, text_type
from simplejson.decoder import JSONDecoder, WHITESPACE


def reference_parse(data):
    """Parse concatenated JSON values with raw_decode over the whole input.

    Returns the same shape as run_stream: ('ok', values, end_offset) or
    ('error', msg, absolute_position, values_decoded_before_the_error).
    """
    decoder = JSONDecoder()
    values = []
    pos = 0
    while True:
        idx = WHITESPACE.match(data, pos).end()
        if idx == len(data):
            return ('ok', values, idx)
        try:
            value, end = decoder.raw_decode(data, idx)
        except JSONDecodeError as err:
            return ('error', err.msg, err.pos, values)
        values.append(value)
        pos = end


def irregular_chunks(data, seed=7, minimum=1, maximum=5):
    rnd = random.Random(seed)
    chunks = []
    i = 0
    while i < len(data):
        n = rnd.randint(minimum, maximum)
        chunks.append(data[i:i + n])
        i += n
    return chunks


def run_stream(chunks, as_bytes=False, **kw):
    """Feed chunks and return (values, offset) or the raised error."""
    decoder = JSONStreamDecoder(**kw)
    values = []
    try:
        for chunk in chunks:
            if as_bytes and isinstance(chunk, text_type):
                chunk = chunk.encode('utf-8')
            values += decoder.feed(chunk)
        values += decoder.eof()
        return ('ok', values, decoder.offset)
    except JSONDecodeError as err:
        # Values decoded from the same chunk before the offending one are
        # carried on the exception so none of the good prefix is lost.
        values += getattr(err, 'values', ())
        return ('error', err.msg, err.pos, values)


class TestJSONStreamDecoder(TestCase):
    values = [1, 2.5, u'€\U0001f600', True, None, [1, 2], {'k': 'v'}]
    good = ' '.join(json.dumps(v) for v in values)

    def test_concatenated_values(self):
        self.assertEqual(run_stream([self.good]),
                         ('ok', self.values, len(self.good)))

    def test_values_returned_incrementally(self):
        decoder = JSONStreamDecoder()
        self.assertEqual(decoder.feed('1 '), [1])
        self.assertEqual(decoder.feed('2.5 '), [2.5])
        self.assertEqual(decoder.feed('"ab'), [])
        self.assertEqual(decoder.feed('c" '), ['abc'])
        self.assertEqual(decoder.eof(), [])

    def test_single_character_feed(self):
        self.assertEqual(run_stream(list(self.good)),
                         ('ok', self.values, len(self.good)))

    def test_single_byte_feed(self):
        encoded = self.good.encode('utf-8')
        chunks = [encoded[i:i + 1] for i in range(len(encoded))]
        self.assertEqual(run_stream(chunks, as_bytes=True),
                         ('ok', self.values, len(self.good)))

    def test_irregular_chunks(self):
        self.assertEqual(run_stream(irregular_chunks(self.good)),
                         ('ok', self.values, len(self.good)))

    def test_irregular_byte_chunks(self):
        chunks = irregular_chunks(self.good.encode('utf-8'), seed=11,
                                  maximum=3)
        self.assertEqual(run_stream(chunks, as_bytes=True),
                         ('ok', self.values, len(self.good)))

    def test_multibyte_utf8_split_across_chunks(self):
        data = u'["€😀", 3]'.encode('utf-8')
        for split in range(1, len(data)):
            decoder = JSONStreamDecoder()
            values = decoder.feed(data[:split])
            values += decoder.feed(data[split:])
            values += decoder.eof()
            self.assertEqual(values, [[u'€😀', 3]])

    def test_backslash_at_chunk_boundary(self):
        # The reported bug: a backslash inside a string landing at the end
        # of a chunk must not make the next chunk look like a new record.
        data = b('"abc\\\\" 42')
        for split in range(1, len(data)):
            decoder = JSONStreamDecoder()
            values = decoder.feed(data[:split])
            values += decoder.feed(data[split:])
            values += decoder.eof()
            self.assertEqual(values, [u'abc\\', 42])

    def test_escape_sequence_split_across_chunks(self):
        decoder = JSONStreamDecoder()
        self.assertEqual(decoder.feed(b'"a\\'), [])
        self.assertEqual(decoder.feed(b'n" 7'), [u'a\n'])
        self.assertEqual(decoder.eof(), [7])

    def test_unicode_escape_split_across_chunks(self):
        decoder = JSONStreamDecoder()
        self.assertEqual(decoder.feed(b'"\\u00'), [])
        self.assertEqual(decoder.feed(b'e9" 7'), [u'é'])
        self.assertEqual(decoder.eof(), [7])

    def test_surrogate_pair_split_across_chunks(self):
        decoder = JSONStreamDecoder()
        self.assertEqual(decoder.feed('"\\ud83d'), [])
        self.assertEqual(decoder.feed('\\ude00" 1'), [u'\U0001f600'])
        self.assertEqual(decoder.eof(), [1])

    def test_number_split_across_chunks(self):
        decoder = JSONStreamDecoder()
        self.assertEqual(decoder.feed('2'), [])
        self.assertEqual(decoder.feed('.5'), [])
        self.assertEqual(decoder.feed('e'), [])
        self.assertEqual(decoder.feed('+'), [])
        # The exponent may still gain digits, so 2.5e3 is only final when
        # a delimiter or EOF proves it.
        self.assertEqual(decoder.feed('3'), [])
        self.assertEqual(decoder.feed(' '), [2500.0])
        self.assertEqual(decoder.eof(), [])

    def test_number_fragment_inside_container(self):
        # A dangling exponent marker or decimal point inside an array must
        # be held rather than reported as a missing delimiter.
        decoder = JSONStreamDecoder()
        self.assertEqual(decoder.feed('[4E'), [])
        self.assertEqual(decoder.feed('6]'), [[4000000.0]])
        self.assertEqual(decoder.eof(), [])

        decoder = JSONStreamDecoder()
        self.assertEqual(decoder.feed('[1, 2.'), [])
        self.assertEqual(decoder.feed('5]'), [[1, 2.5]])
        self.assertEqual(decoder.eof(), [])

    def test_hard_error_inside_container_keeps_position(self):
        decoder = JSONStreamDecoder()
        try:
            decoder.feed('[4E6{')
        except JSONDecodeError as err:
            self.assertEqual(err.msg,
                             "Expecting ',' delimiter or ']'")
            self.assertEqual(err.pos, 4)
        else:
            self.fail('JSONDecodeError not raised')

    def test_root_numbers(self):
        for text, value in [('42', 42), ('-17', -17), ('3.5', 3.5),
                            ('1e10', 1e10), ('0', 0)]:
            self.assertEqual(run_stream([text]), ('ok', [value], len(text)))

    def test_root_number_needs_eof(self):
        # A bare integer may still gain a fraction or exponent, so it is
        # only emitted once the stream is closed.
        decoder = JSONStreamDecoder()
        self.assertEqual(decoder.feed('42'), [])
        self.assertEqual(decoder.eof(), [42])

    def test_whitespace_only_tail(self):
        self.assertEqual(run_stream(['123   ']), ('ok', [123], 6))
        self.assertEqual(run_stream(['   ']), ('ok', [], 3))
        self.assertEqual(run_stream([' \t\n']), ('ok', [], 3))
        self.assertEqual(run_stream([]), ('ok', [], 0))

    def test_eof_with_final_chunk(self):
        decoder = JSONStreamDecoder()
        self.assertEqual(decoder.feed('1 '), [1])
        self.assertEqual(decoder.eof('2 '), [2])

    def test_decode_final_interface(self):
        decoder = JSONStreamDecoder()
        self.assertEqual(decoder.decode('1 '), [1])
        self.assertEqual(decoder.decode(b'2.5 ', final=True), [2.5])
        self.assertTrue(decoder.closed)

    def test_close_is_eof(self):
        decoder = JSONStreamDecoder()
        self.assertEqual(decoder.feed('1 '), [1])
        self.assertEqual(decoder.close('2'), [2])
        self.assertTrue(decoder.closed)

    def test_bad_tail_error_position(self):
        data = self.good + ' xyz'
        # Character by character: the complete prefix is returned before
        # the bad tail is rejected.
        result = run_stream(list(data))
        self.assertEqual(result[:3],
                         ('error', 'Expecting value', len(self.good) + 1))
        self.assertEqual(result[3], self.values)

    def test_bad_tail_parity(self):
        # Feed the same input byte by byte and in irregular chunks: the
        # values from the good prefix and the error position of the bad
        # tail must agree with each other and with a plain raw_decode parse
        # of the whole input.
        tails = ['xyz', 'tru', '"unterminated', '[1,', '{"a":', '"\\u12',
                 '-', '{"a" 1}', '[1,,2]', '"\\q"', '12a', '123.', '2.5e+',
                 '[1 2]', '[1,]', '{"a":"\\uXYZ"}']
        for tail in tails:
            data = self.good + ' ' + tail
            expected = reference_parse(data)
            runs = [
                run_stream(list(data)),
                run_stream([data.encode('utf-8')[i:i + 1]
                            for i in range(len(data.encode('utf-8')))],
                           as_bytes=True),
                run_stream(irregular_chunks(data)),
                run_stream(irregular_chunks(data.encode('utf-8'), seed=11,
                                            maximum=3), as_bytes=True),
            ]
            for result in runs:
                self.assertEqual(result, expected, tail)

    def test_error_positions_are_absolute(self):
        decoder = JSONStreamDecoder()
        self.assertEqual(decoder.feed('1\n2\n'), [1, 2])
        try:
            decoder.eof('bad')
        except JSONDecodeError as err:
            self.assertEqual(err.pos, 4)
            self.assertEqual((err.lineno, err.colno), (2, 1))
        else:
            self.fail('JSONDecodeError not raised')

    def test_hard_error_raised_by_feed(self):
        decoder = JSONStreamDecoder()
        try:
            decoder.feed('{"a" 1}')
        except JSONDecodeError as err:
            self.assertEqual(err.msg, "Expecting ':' delimiter")
            self.assertEqual(err.pos, 5)
        else:
            self.fail('JSONDecodeError not raised')

    def test_incomplete_value_at_eof(self):
        cases = [
            ('{"a": [1, 2', "Expecting ',' delimiter or ']'", 11),
            ('tru', 'Expecting value', 0),
            ('"abc\\', 'Unterminated string starting at', 0),
            ('[1,', 'Expecting value', 3),
            ('{"a":', 'Expecting value', 5),
            ('"\\u12', 'Invalid \\uXXXX escape sequence', 1),
            ('-', 'Expecting value', 0),
        ]
        for data, msg, pos in cases:
            result = run_stream([data])
            self.assertEqual(result, ('error', msg, pos, []), data)

    def test_invalid_escape_not_repaired(self):
        # A complete but non-hex escape is invalid no matter what follows.
        decoder = JSONStreamDecoder()
        try:
            decoder.feed('"\\uXY')
        except JSONDecodeError as err:
            self.assertEqual(err.msg, 'Invalid \\uXXXX escape sequence')
            self.assertEqual(err.pos, 1)
        else:
            self.fail('JSONDecodeError not raised')

    def test_decoder_options(self):
        self.assertEqual(
            run_stream(['1.5 2.5'], parse_float=Decimal),
            ('ok', [Decimal('1.5'), Decimal('2.5')], 7))
        self.assertEqual(
            run_stream(['{"a":1}'], object_hook=lambda o: ('dict', o)),
            ('ok', [('dict', {'a': 1})], 7))
        self.assertEqual(
            run_stream(['{"a":1}'], object_pairs_hook=list),
            ('ok', [[('a', 1)]], 7))
        self.assertEqual(
            run_stream(['[1,2]'], array_hook=tuple),
            ('ok', [(1, 2)], 5))
        self.assertEqual(
            run_stream(['"a\x01b"'], strict=False),
            ('ok', [u'a\x01b'], 5))
        result = run_stream(['NaN Infinity -Infinity'], allow_nan=True)
        self.assertEqual(result[0], 'ok')
        self.assertEqual(result[2], 22)
        self.assertTrue(result[1][0] != result[1][0])  # NaN
        self.assertEqual(result[1][1:], [float('inf'), float('-inf')])

    def test_custom_decoder_class(self):
        class MyDecoder(JSONDecoder):
            pass
        decoder = JSONStreamDecoder(cls=MyDecoder)
        self.assertIsInstance(decoder.decoder, MyDecoder)
        self.assertEqual(decoder.eof('1'), [1])

    def test_decoder_instance(self):
        decoder = JSONStreamDecoder(decoder=JSONDecoder(parse_int=float))
        self.assertEqual(decoder.eof('1 2'), [1.0, 2.0])
        self.assertRaises(TypeError, JSONStreamDecoder,
                          decoder=JSONDecoder(), strict=False)
        self.assertRaises(TypeError, JSONStreamDecoder, decoder='nope')

    def test_encoding_option(self):
        decoder = JSONStreamDecoder(encoding='latin1')
        self.assertEqual(decoder.eof(b'"\xe9"'), [u'\xe9'])

    def test_bom_at_start(self):
        self.assertEqual(run_stream([u'﻿[1] 2']), ('ok', [[1], 2], 6))

    def test_mixed_text_and_bytes(self):
        decoder = JSONStreamDecoder()
        self.assertEqual(decoder.feed('[1,'), [])
        self.assertEqual(decoder.feed(b'2] '), [[1, 2]])
        self.assertEqual(decoder.eof(), [])

    def test_empty_chunks(self):
        decoder = JSONStreamDecoder()
        self.assertEqual(decoder.feed(''), [])
        self.assertEqual(decoder.feed(b''), [])
        self.assertEqual(decoder.feed('1'), [])
        self.assertEqual(decoder.eof(), [1])

    def test_feed_after_eof(self):
        decoder = JSONStreamDecoder()
        decoder.eof()
        self.assertRaises(ValueError, decoder.feed, '1')
        self.assertRaises(ValueError, decoder.eof)

    def test_bad_chunk_type(self):
        decoder = JSONStreamDecoder()
        self.assertRaises(TypeError, decoder.feed, 123)
        self.assertRaises(TypeError, decoder.eof, 123)

    def test_offset_property(self):
        decoder = JSONStreamDecoder()
        self.assertEqual(decoder.offset, 0)
        decoder.feed('1 ')
        self.assertEqual(decoder.offset, 1)
        decoder.feed('2.5 ')
        self.assertEqual(decoder.offset, 5)
        decoder.eof()
        self.assertEqual(decoder.offset, 6)

    def test_truncated_utf8_at_eof(self):
        decoder = JSONStreamDecoder()
        decoder.feed(b'["\xe2\x82')
        self.assertRaises(UnicodeDecodeError, decoder.eof)
