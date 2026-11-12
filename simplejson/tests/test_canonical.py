# -*- coding: utf-8 -*-
from __future__ import absolute_import

import decimal
import unittest
from decimal import Decimal

import simplejson as json
from simplejson import RawJSON
from simplejson.compat import StringIO
from simplejson.encoder import JSONEncoder


class TestCanonical(unittest.TestCase):
    def dumps(self, obj, **kw):
        sio = StringIO()
        json.dump(obj, sio, canonical=True, **kw)
        res = json.dumps(obj, canonical=True, **kw)
        self.assertEqual(res, sio.getvalue())
        return res

    def test_equivalent_objects_share_content_key(self):
        # Two producers: Decimal('1.00') vs int 1, different insertion
        # order. Their UTF-8 content keys must be identical.
        producer_a = {
            'amount': Decimal('1.00'),
            'items': [{'name': 'x', 'qty': Decimal('2.50')},
                      {'name': 'y', 'qty': 3}],
            'meta': {'created': 1000000, 'flag': True, 'delta': -0.0},
        }
        producer_b = {
            'meta': {'delta': float('-0.0'), 'flag': True,
                     'created': 1000000},
            'items': [{'qty': 2.5, 'name': 'x'},
                      {'qty': 3, 'name': 'y'}],
            'amount': 1,
        }
        key_a = self.dumps(producer_a)
        key_b = self.dumps(producer_b)
        self.assertEqual(key_a, key_b)
        self.assertEqual(
            key_a.encode('utf-8'), key_b.encode('utf-8'))
        self.assertEqual(
            key_a,
            '{"amount":1,"items":[{"name":"x","qty":2.5},'
            '{"name":"y","qty":3}],"meta":{'
            '"created":1000000,"delta":0,"flag":true}}')

    def test_content_key_is_stable_over_reorderings(self):
        base = {
            'z': [1, {'k': Decimal('3.00'), 'j': -0}],
            'a': {'y': 1.0, 'x': Decimal('0.500')},
            'm': None,
        }
        # Rebuilt in a deliberately different insertion order and with
        # int/float/Decimal swapped for equal numeric values.
        reordered = {
            'm': None,
            'a': {'x': Decimal('0.5'), 'y': 1},
            'z': [1.0, {'j': float('-0.0'), 'k': Decimal('3')}],
        }
        self.assertEqual(self.dumps(base), self.dumps(reordered))

    def test_high_precision_decimal_does_not_collide_with_float(self):
        precise = Decimal('0.12345678901234567890123456789012345678')
        approximate = float(precise)
        key_precise = self.dumps({'v': precise})
        key_approx = self.dumps({'v': approximate})
        self.assertNotEqual(key_precise, key_approx)
        self.assertNotEqual(
            key_precise.encode('utf-8'), key_approx.encode('utf-8'))
        self.assertEqual(
            key_precise,
            '{"v":0.12345678901234567890123456789012345678}')
        self.assertEqual(
            key_approx, '{"v":0.12345678901234568}')

    def test_decimal_precision_is_preserved(self):
        self.assertEqual(
            self.dumps(Decimal('1234567890.1234567890')),
            '1234567890.123456789')
        # normalizing must not introduce exponent notation into the output
        self.assertEqual(
            self.dumps(Decimal('1E+30')),
            '1000000000000000000000000000000')
        self.assertEqual(
            self.dumps(Decimal('0.0000000000000000000001')),
            '0.0000000000000000000001')

    def test_decimal_precision_beyond_context_is_preserved(self):
        # Decimal.normalize() rounds to the active context precision;
        # canonical output must not. Verify digits beyond the default
        # 28-digit context survive, and that a poisoned global context
        # cannot change the result.
        precise = Decimal('1.' + '1' * 99)
        self.assertEqual(self.dumps(precise), '1.' + '1' * 99)
        with decimal.localcontext() as ctx:
            ctx.prec = 3
            self.assertEqual(
                self.dumps(Decimal('1234567890.1234567890')),
                '1234567890.123456789')
            self.assertEqual(self.dumps(1.5), '1.5')

    def test_negative_zero_canonicalizes_to_zero(self):
        for value in (-0, float('-0.0'), Decimal('-0.00'),
                      Decimal('-0')):
            self.assertEqual(self.dumps(value), '0')
            self.assertEqual(self.dumps([value]), '[0]')
            self.assertEqual(self.dumps({'k': value}), '{"k":0}')

    def test_equal_numbers_share_one_representation(self):
        for value in (1, 1.0, Decimal('1'), Decimal('1.0'),
                      Decimal('1.00')):
            self.assertEqual(self.dumps(value), '1')
        for value in (25, 25.0, Decimal('25.000')):
            self.assertEqual(self.dumps(value), '25')
        for value in (0, 0.0, Decimal('0.000')):
            self.assertEqual(self.dumps(value), '0')

    def test_big_integers_are_not_stringified(self):
        self.assertEqual(
            self.dumps({'n': 10 ** 40}),
            '{"n":10000000000000000000000000000000000000000}')

    def test_non_finite_values_are_rejected(self):
        non_finite = (
            float('nan'), float('inf'), float('-inf'),
            Decimal('NaN'), Decimal('sNaN'),
            Decimal('Infinity'), Decimal('-Infinity'),
        )
        for value in non_finite:
            for obj in (value, [value], {'k': value}):
                self.assertRaises(
                    ValueError, json.dumps, obj, canonical=True)

    def test_allow_nan_and_ignore_nan_cannot_override_rejection(self):
        for kwargs in (dict(allow_nan=True), dict(ignore_nan=True)):
            self.assertRaises(
                ValueError, json.dumps, float('nan'),
                canonical=True, **kwargs)
            self.assertRaises(
                ValueError, json.dumps, Decimal('Infinity'),
                canonical=True, **kwargs)

    def test_non_string_keys_are_rejected(self):
        for key in (1, 1.5, None, True, Decimal('1'), float('nan')):
            self.assertRaises(
                TypeError, json.dumps, {key: 1}, canonical=True)

    def test_bytes_keys_are_decoded_then_checked(self):
        # bytes keys are only allowed if decoding them with the
        # configured encoding yields a string
        self.assertEqual(
            self.dumps({b'k\xc3\xa9': 1}),
            '{"k\\u00e9":1}')
        self.assertRaises(
            TypeError, json.dumps, {b'k': 1},
            canonical=True, encoding=None)

    def test_duplicate_keys_after_decoding_are_rejected(self):
        # b'k' decodes to 'k': emitting both would produce a document
        # whose content key is ambiguous
        self.assertRaises(
            ValueError, json.dumps, {'k': 1, b'k': 2}, canonical=True)
        self.assertEqual(
            self.dumps({'k': 1, b'j': 2}), '{"j":2,"k":1}')

    def test_raw_json_is_rejected_everywhere(self):
        documents = (
            RawJSON('{"x":1}'),
            [RawJSON('1')],
            {'k': RawJSON('true')},
        )
        for document in documents:
            self.assertRaises(
                TypeError, json.dumps, document, canonical=True)
            self.assertRaises(
                TypeError,
                lambda doc=document: ''.join(
                    JSONEncoder(canonical=True).iterencode(doc)))

    def test_fixed_string_escaping(self):
        # ASCII-only output: non-ASCII, DEL and control chars are
        # escaped with lowercase \uXXXX; short escapes stay short.
        text = u'héllo\x7f\x01\t\U0001f600'
        encoded = self.dumps(text)
        self.assertEqual(
            encoded,
            '"h\\u00e9llo\\u007f\\u0001\\t'
            '\\ud83d\\ude00"')
        # the UTF-8 content key is pure ASCII here
        encoded.encode('ascii')
        # the JSONEncoder.encode() fast path for top-level strings must
        # use the same canonical escaper
        self.assertEqual(
            JSONEncoder(canonical=True).encode(text), encoded)

    def test_fixed_whitespace_and_key_order(self):
        encoded = self.dumps({'b': 1, 'a': [1, 2], 'c': None})
        self.assertEqual(encoded, '{"a":[1,2],"b":1,"c":null}')
        self.assertNotIn(' ', encoded)

    def test_incompatible_arguments_are_rejected(self):
        bad_kwargs = (
            dict(indent=2),
            dict(separators=(',', ':')),
            dict(item_sort_key=lambda kv: kv[0]),
            dict(skipkeys=True),
            dict(bigint_as_string=True),
            dict(int_as_string_bitcount=32),
            dict(allow_nan=True),
            dict(ignore_nan=True),
            dict(use_decimal=False),
        )
        for kwargs in bad_kwargs:
            self.assertRaises(
                ValueError, JSONEncoder, canonical=True, **kwargs)
            self.assertRaises(
                ValueError, json.dumps, 1, canonical=True, **kwargs)

    def test_encoder_attributes_are_pinned(self):
        encoder = JSONEncoder(canonical=True)
        self.assertTrue(encoder.canonical)
        self.assertTrue(encoder.sort_keys)
        self.assertTrue(encoder.ensure_ascii)
        self.assertEqual(encoder.item_separator, ',')
        self.assertEqual(encoder.key_separator, ':')

    def test_uses_pure_python_encoder(self):
        # The canonical output must not depend on whether the C
        # extension is available; this assertion pins the behavior in
        # both the speedups and NoExtension test runs.
        before = json.encoder.c_make_encoder
        try:
            json.encoder.c_make_encoder = None
            encoded = json.dumps({'b': Decimal('1.00'), 'a': 1},
                                 canonical=True)
        finally:
            json.encoder.c_make_encoder = before
        self.assertEqual(encoded, '{"a":1,"b":1}')

    def test_defaults_are_unchanged(self):
        self.assertEqual(
            json.dumps({'b': 1, 'a': 2}), '{"b": 1, "a": 2}')
        self.assertEqual(json.dumps(Decimal('1.00')), '1.00')
        self.assertEqual(json.dumps(float('-0.0')), '-0.0')
        self.assertEqual(json.dumps(float('nan'), ignore_nan=True),
                         'null')
        self.assertEqual(
            json.dumps(RawJSON('{"x":1}')), '{"x":1}')
        self.assertEqual(
            json.dumps({'b': 1, 'a': 2}, sort_keys=True),
            '{"a": 2, "b": 1}')
        self.assertEqual(JSONEncoder().canonical, False)


if __name__ == '__main__':
    unittest.main()
