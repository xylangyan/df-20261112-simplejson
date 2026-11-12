from __future__ import absolute_import
import types
from unittest import TestCase

import simplejson as json
from simplejson.compat import StringIO, b


class CountingIterator(object):
    """One-shot iterator that records how many items were pulled."""
    def __init__(self, values):
        self.values = list(values)
        self.reads = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.reads >= len(self.values):
            raise StopIteration
        value = self.values[self.reads]
        self.reads += 1
        return value

    next = __next__


class OneShotIterable(object):
    """An iterable that can only be iterated once, like the small
    collections nested inside a row stream."""
    def __init__(self, values):
        self.values = list(values)
        self.iterations = 0

    def __iter__(self):
        self.iterations += 1
        if self.iterations > 1:
            raise AssertionError("iterated more than once")
        return iter(self.values)


class RecordingIO(object):
    """Minimal write()-supporting sink that records each chunk. With
    ``fail_after`` set, write() raises once that many chunks are in,
    modeling a consumer (socket, pipe) that breaks the loop early."""
    def __init__(self, fail_after=None):
        self.parts = []
        self.fail_after = fail_after
        self.writes = 0

    def write(self, s):
        if self.fail_after is not None and self.writes >= self.fail_after:
            raise IOError("sink closed")
        self.writes += 1
        self.parts.append(s)
        return len(s)

    def getvalue(self):
        return ''.join(self.parts)


def make_rows():
    """Rows produced one at a time, each holding a one-shot collection."""
    return (OneShotIterable([i, i + 1]) for i in range(3))


class TestIterencodeStreaming(TestCase):
    def test_iterencode_returns_a_lazy_generator(self):
        source = CountingIterator(range(10))
        chunks = json.JSONEncoder(iterable_as_array=True).iterencode(source)
        self.assertIsInstance(chunks, types.GeneratorType)
        # Nothing is pulled from the source before the first chunk is read.
        self.assertEqual(source.reads, 0)
        first = next(chunks)
        self.assertEqual(first, '[0')
        self.assertEqual(source.reads, 1)

    def test_first_chunks_after_few_reads(self):
        source = CountingIterator(range(10))
        chunks = iter(json.JSONEncoder(iterable_as_array=True)
                      .iterencode(source))
        received = [next(chunks) for _ in range(3)]
        self.assertEqual(received, ['[0', ', 1', ', 2'])
        self.assertEqual(source.reads, 3)

    def test_abandoned_generator_stops_pulling(self):
        source = CountingIterator(range(10))
        chunks = iter(json.JSONEncoder(iterable_as_array=True)
                      .iterencode(source))
        for _ in range(3):
            next(chunks)
        self.assertEqual(source.reads, 3)
        # Stop reading: the source iterator is not pulled again.
        del chunks
        self.assertEqual(source.reads, 3)

    def test_nested_one_shot_iterables(self):
        rows = make_rows()
        chunks = json.JSONEncoder(iterable_as_array=True).iterencode(rows)
        self.assertEqual(''.join(chunks), '[[0, 1], [1, 2], [2, 3]]')

    def test_dump_writes_incrementally(self):
        source = CountingIterator(range(8))
        fp = RecordingIO()
        json.dump(source, fp, iterable_as_array=True)
        # The document is written piece by piece, not as one giant string.
        self.assertGreater(fp.writes, 1)
        self.assertEqual(fp.getvalue(), '[0, 1, 2, 3, 4, 5, 6, 7]')
        self.assertEqual(json.loads(fp.getvalue()), list(range(8)))

    def test_first_chunk_can_be_written_after_one_pull(self):
        source = CountingIterator(range(8))
        fp = RecordingIO()
        chunks = json.JSONEncoder(iterable_as_array=True).iterencode(source)
        fp.write(next(chunks))
        self.assertEqual(source.reads, 1)
        self.assertEqual(fp.getvalue(), '[0')

    def test_default_callback_count_matches_dumps(self):
        class Holder(object):
            def __init__(self, value):
                self.value = value

        calls = []

        def default(obj):
            if isinstance(obj, Holder):
                calls.append(obj.value)
                return obj.value
            raise TypeError(repr(obj))

        data = [Holder(1), [Holder(2)], {'h': Holder(3)}, Holder(4)]
        expected = json.dumps(data, default=default)
        self.assertEqual(calls, [1, 2, 3, 4])
        del calls[:]
        streamed = ''.join(
            json.JSONEncoder(default=default).iterencode(data))
        self.assertEqual(calls, [1, 2, 3, 4])
        self.assertEqual(streamed, expected)

    def test_full_document_matches_dumps(self):
        data = [1, 'two', 3.0, True, False, None, [4, 5],
                {'k': 'v', 'n': [None]}, 'end']
        for kw in ({}, {'indent': 2}, {'indent': '\t'},
                   {'sort_keys': True, 'indent': 2},
                   {'separators': (',', ':')},
                   {'ensure_ascii': False}):
            expected = json.dumps(data, **kw)
            streamed = ''.join(json.JSONEncoder(**kw).iterencode(data))
            self.assertEqual(streamed, expected)
            self.assertEqual(json.loads(streamed), json.loads(expected))

    def test_encoding_option_matches_dumps(self):
        data = [b('\xa4')]
        expected = json.dumps(data, encoding='iso-8859-1')
        streamed = ''.join(
            json.JSONEncoder(encoding='iso-8859-1').iterencode(data))
        self.assertEqual(streamed, expected)

    def test_circular_reference_detected(self):
        dct = {}
        dct['self'] = dct
        chunks = json.JSONEncoder().iterencode(dct)
        self.assertRaises(ValueError, list, chunks)

        lst = []
        lst.append(lst)
        chunks = json.JSONEncoder().iterencode(lst)
        self.assertRaises(ValueError, list, chunks)

    def test_circular_iterable_detected(self):
        class SelfIterable(object):
            def __iter__(self):
                return iter([self])

        chunks = json.JSONEncoder(iterable_as_array=True).iterencode(
            SelfIterable())
        self.assertRaises(ValueError, list, chunks)

    def test_mid_stream_error_leaves_incomplete_output(self):
        class Boom(object):
            def __iter__(self):
                yield 1
                yield object()

        fp = StringIO()
        self.assertRaises(TypeError, json.dump, Boom(), fp,
                          iterable_as_array=True)
        partial = fp.getvalue()
        # The partial document must not look like a complete array.
        self.assertFalse(partial.rstrip().endswith(']'), repr(partial))
        self.assertRaises(ValueError, json.loads, partial)

    def test_error_from_source_iterator_propagates(self):
        class Failing(object):
            def __iter__(self):
                yield 1
                raise RuntimeError('source failed')

        fp = StringIO()
        self.assertRaises(RuntimeError, json.dump, Failing(), fp,
                          iterable_as_array=True)
        self.assertFalse(fp.getvalue().rstrip().endswith(']'))

    def test_failing_sink_stops_consuming_the_source(self):
        source = CountingIterator(range(10))
        fp = RecordingIO(fail_after=2)
        self.assertRaises(IOError, json.dump, source, fp,
                          iterable_as_array=True)
        # The sink died while writing the third chunk; at most one more
        # item than successful writes was pulled, and the remaining
        # items were never touched.
        self.assertLessEqual(source.reads, fp.writes + 1)
        self.assertLess(source.reads, len(source.values))

    def test_empty_iterable_streams_compactly(self):
        source = CountingIterator([])
        fp = RecordingIO()
        json.dump(source, fp, iterable_as_array=True)
        self.assertEqual(fp.getvalue(), '[]')
        self.assertEqual(fp.writes, 1)
        self.assertEqual(source.reads, 0)
        for kw in ({}, {'indent': 2}):
            streamed = ''.join(
                json.JSONEncoder(iterable_as_array=True, **kw)
                .iterencode(CountingIterator([])))
            self.assertEqual(streamed, '[]')

    def test_for_html_iterencode_is_lazy(self):
        source = CountingIterator(range(5))
        chunks = json.JSONEncoderForHTML(iterable_as_array=True).iterencode(
            source)
        self.assertIsInstance(chunks, types.GeneratorType)
        first = next(chunks)
        self.assertEqual(first, '[0')
        self.assertEqual(source.reads, 1)
        self.assertEqual(''.join([first] + list(chunks)),
                         '[0, 1, 2, 3, 4]')


class TestOneShotEncoding(TestCase):
    def test_one_shot_and_lazy_paths_agree(self):
        data = [1, 'two', 3.0, True, None, [4, 5], {'k': 'v'}]
        for kw in ({}, {'indent': 2}, {'sort_keys': True},
                   {'ensure_ascii': False}):
            encoder_ = json.JSONEncoder(**kw)
            lazy = ''.join(encoder_.iterencode(data))
            one_shot = encoder_.iterencode(data, _one_shot=True)
            if not isinstance(one_shot, (list, tuple)):
                one_shot = list(one_shot)
            self.assertEqual(''.join(one_shot), lazy)
            self.assertEqual(encoder_.encode(data), lazy)
            self.assertEqual(json.dumps(data, **kw), lazy)

    def test_dumps_matches_iterencode(self):
        data = {'rows': [[0, 1], [1, 2]], 'done': False}
        streamed = ''.join(json.JSONEncoder().iterencode(data))
        self.assertEqual(json.dumps(data), streamed)
