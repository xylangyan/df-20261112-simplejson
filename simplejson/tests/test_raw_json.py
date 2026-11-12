import unittest
import simplejson as json

dct1 = {
    'key1': 'value1'
}

dct2 = {
    'key2': 'value2',
    'd1': dct1
}

dct3 = {
    'key2': 'value2',
    'd1': json.dumps(dct1)
}

dct4 = {
    'key2': 'value2',
    'd1': json.RawJSON(json.dumps(dct1))
}


class TestRawJson(unittest.TestCase):

    def test_normal_str(self):
        self.assertNotEqual(json.dumps(dct2), json.dumps(dct3))

    def test_raw_json_str(self):
        self.assertEqual(json.dumps(dct2), json.dumps(dct4))
        self.assertEqual(dct2, json.loads(json.dumps(dct4)))

    def test_list(self):
        self.assertEqual(
            json.dumps([dct2]),
            json.dumps([json.RawJSON(json.dumps(dct2))]))
        self.assertEqual(
            [dct2],
            json.loads(json.dumps([json.RawJSON(json.dumps(dct2))])))

    def test_direct(self):
        self.assertEqual(
            json.dumps(dct2),
            json.dumps(json.RawJSON(json.dumps(dct2))))
        self.assertEqual(
            dct2,
            json.loads(json.dumps(json.RawJSON(json.dumps(dct2)))))


class TestReformatRawJson(unittest.TestCase):
    """reformat_raw_json re-encodes RawJSON fragments to match the
    surrounding document instead of embedding them byte-for-byte."""

    def _check_compact_and_pretty(self, fragment, compact, pretty):
        doc = {'archive': json.RawJSON(fragment)}
        self.assertEqual(
            json.dumps(doc, separators=(',', ':'), reformat_raw_json=True),
            compact)
        self.assertEqual(
            json.dumps(doc, indent=2, reformat_raw_json=True),
            pretty)

    def test_nested_arrays(self):
        self._check_compact_and_pretty(
            '[[1, [2, 3]], {"k": [true, null]}]',
            '{"archive":[[1,[2,3]],{"k":[true,null]}]}',
            '{\n'
            '  "archive": [\n'
            '    [\n'
            '      1,\n'
            '      [\n'
            '        2,\n'
            '        3\n'
            '      ]\n'
            '    ],\n'
            '    {\n'
            '      "k": [\n'
            '        true,\n'
            '        null\n'
            '      ]\n'
            '    }\n'
            '  ]\n'
            '}')

    def test_quoted_string(self):
        self._check_compact_and_pretty(
            '{"quote": "a \\"quoted\\" value"}',
            '{"archive":{"quote":"a \\"quoted\\" value"}}',
            '{\n'
            '  "archive": {\n'
            '    "quote": "a \\"quoted\\" value"\n'
            '  }\n'
            '}')

    def test_trailing_whitespace(self):
        self._check_compact_and_pretty(
            '{"note": "done"}  \n\t',
            '{"archive":{"note":"done"}}',
            '{\n'
            '  "archive": {\n'
            '    "note": "done"\n'
            '  }\n'
            '}')

    def test_default_is_passthrough(self):
        # Without reformat_raw_json the fragment bytes are embedded
        # verbatim, including any trailing newline.
        fragment = '{"a": 1}\n'
        self.assertEqual(
            json.dumps({'archive': json.RawJSON(fragment)}),
            '{"archive": {"a": 1}\n}')
        self.assertEqual(
            json.dumps({'archive': json.RawJSON(fragment)}, indent=2),
            '{\n  "archive": {"a": 1}\n\n}')

    def test_top_level_fragment(self):
        self.assertEqual(
            json.dumps(json.RawJSON('[1, 2]'), reformat_raw_json=True),
            '[1, 2]')
        self.assertEqual(
            json.dumps(json.RawJSON('{"a": 1}'), indent=2,
                       reformat_raw_json=True),
            '{\n  "a": 1\n}')

    def test_matches_normal_encoding(self):
        # Reformatting a fragment must produce the same bytes as encoding
        # the equivalent Python value directly.
        fragment = '[[1, [2, 3]], {"k": [true, null]}]'
        value = json.loads(fragment)
        for kw in ({'separators': (',', ':')}, {'indent': 2},
                   {'ensure_ascii': False, 'indent': '\t'}):
            self.assertEqual(
                json.dumps({'archive': json.RawJSON(fragment)},
                           reformat_raw_json=True, **kw),
                json.dumps({'archive': value}, **kw))

    def test_ensure_ascii_rules_apply(self):
        fragment = '{"s": "caf\\u00e9"}'
        self.assertEqual(
            json.dumps(json.RawJSON(fragment), reformat_raw_json=True),
            '{"s": "caf\\u00e9"}')
        self.assertEqual(
            json.dumps(json.RawJSON(fragment), ensure_ascii=False,
                       reformat_raw_json=True),
            u'{"s": "caf\xe9"}')

    def test_extra_data_points_at_fragment(self):
        fragment = '[1, 2] junk'
        with self.assertRaises(json.JSONDecodeError) as cm:
            json.dumps({'archive': json.RawJSON(fragment)},
                       reformat_raw_json=True)
        err = cm.exception
        self.assertEqual(err.doc, fragment)
        self.assertEqual(err.pos, fragment.index('j'))

    def test_invalid_fragment_points_at_fragment(self):
        fragment = '{"a": }'
        with self.assertRaises(json.JSONDecodeError) as cm:
            json.dumps({'archive': json.RawJSON(fragment)},
                       reformat_raw_json=True)
        self.assertEqual(cm.exception.doc, fragment)

    def test_non_str_fragment_raises(self):
        with self.assertRaises(TypeError):
            json.dumps(json.RawJSON(42), reformat_raw_json=True)

    def test_out_of_range_float_in_fragment(self):
        # 1e400 decodes to inf; reformatting follows the encoder's
        # allow_nan/ignore_nan rules instead of emitting invalid JSON.
        with self.assertRaises(ValueError):
            json.dumps(json.RawJSON('1e400'), reformat_raw_json=True)
        self.assertEqual(
            json.dumps(json.RawJSON('1e400'), allow_nan=True,
                       reformat_raw_json=True),
            'Infinity')
        self.assertEqual(
            json.dumps(json.RawJSON('1e400'), ignore_nan=True,
                       reformat_raw_json=True),
            'null')

    def test_iterencode_raises_before_partial_output(self):
        # A bad fragment aborts the encode; no partial document is
        # produced once the error is raised. The C encoder raises from
        # iterencode() itself, the pure Python encoder from iteration.
        doc = {'archive': json.RawJSON('{'), 'other': 1}
        with self.assertRaises(json.JSONDecodeError):
            list(json.JSONEncoder(reformat_raw_json=True).iterencode(doc))
