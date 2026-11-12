from __future__ import absolute_import
from unittest import TestCase

import simplejson as json


def make_field(name):
    return ''.join([name[:1], name[1:]])


class RecordingPairsHook(object):
    def __init__(self):
        self.calls = []

    def __call__(self, pairs):
        self.calls.append(pairs)
        return dict(pairs)


class TestReusableKeys(TestCase):
    def setUp(self):
        self.device = make_field('device')
        self.temp = make_field('temp')
        self.other = make_field('other')
        self.keys = [self.device, self.temp]

    def test_three_reads_reuse_only_declared_keys(self):
        decoder = json.JSONDecoder(reusable_keys=self.keys)
        documents = [
            u'{"device": 1, "temp": 2, "other": 3}',
            u'{"device": 4, "temp": 5, "other": 6}',
            u'{"device": 7, "temp": 8, "other": 9}',
        ]
        results = [decoder.decode(document) for document in documents]
        reused_keys = {
            'device': [list(result)[0] for result in results],
            'temp': [list(result)[1] for result in results],
        }
        independent_keys = [list(result)[2] for result in results]

        for result in results:
            device_key, temp_key, other_key = list(result)
            self.assertIs(device_key, self.device)
            self.assertIs(temp_key, self.temp)
            self.assertIsNot(other_key, self.device)
            self.assertIsNot(other_key, self.temp)
        self.assertEqual(reused_keys['device'], [self.device] * 3)
        self.assertEqual(reused_keys['temp'], [self.temp] * 3)
        self.assertEqual(len(set(map(id, independent_keys))), 3)

    def test_nested_objects_use_the_same_table(self):
        decoder = json.JSONDecoder(reusable_keys=self.keys)
        result = decoder.decode(
            u'{"device": {"temp": {"device": 1}}, "temp": 2}')
        self.assertIs(list(result)[0], self.device)
        self.assertIs(list(result)[1], self.temp)
        nested = result[self.device]
        self.assertIs(list(nested)[0], self.temp)
        self.assertIs(list(nested[self.temp])[0], self.device)
        self.assertEqual(result, {'device': {'temp': {'device': 1}},
                                  'temp': 2})

    def test_escaped_key_uses_declared_identity(self):
        decoder = json.JSONDecoder(reusable_keys=self.keys)
        result = decoder.decode(u'{"dev\\u0069ce": 1, "te\\u006d\\u0070": 2}')
        self.assertIs(list(result)[0], self.device)
        self.assertIs(list(result)[1], self.temp)

    def test_object_pairs_hook_sees_reused_keys(self):
        hook = RecordingPairsHook()
        decoder = json.JSONDecoder(
            reusable_keys=self.keys, object_pairs_hook=hook)
        decoder.decode(u'{"device": {"temp": 1}, "other": 2}')
        inner, outer = hook.calls
        self.assertEqual([key for key, value in outer],
                         ['device', 'other'])
        self.assertIs(outer[0][0], self.device)
        self.assertIs(inner[0][0], self.temp)
        self.assertIsNot(outer[1][0], self.device)

    def test_object_hook_sees_reused_keys(self):
        seen = []
        decoder = json.JSONDecoder(
            reusable_keys=self.keys,
            object_hook=lambda value: seen.append(value) or value)
        decoder.decode(u'{"device": {"temp": 1}}')
        inner, outer = seen
        self.assertIs(list(inner)[0], self.temp)
        self.assertIs(list(outer)[0], self.device)

    def test_string_values_are_not_reused(self):
        decoder = json.JSONDecoder(reusable_keys=self.keys)
        first = decoder.decode(u'{"device": "device"}')
        second = decoder.decode(u'{"device": "device"}')
        self.assertIs(list(first)[0], self.device)
        self.assertIs(list(second)[0], self.device)
        self.assertEqual(first['device'], 'device')
        self.assertIsNot(first['device'], self.device)
        self.assertIsNot(first['device'], second['device'])

    def test_decoders_have_independent_tables(self):
        first_key = make_field('device')
        second_key = make_field('device')
        self.assertIsNot(first_key, second_key)
        first = json.JSONDecoder(reusable_keys=[first_key])
        second = json.JSONDecoder(reusable_keys=[second_key])
        first_result = first.decode(u'{"device": 1}')
        second_result = second.decode(u'{"device": 2}')
        self.assertIs(list(first_result)[0], first_key)
        self.assertIs(list(second_result)[0], second_key)
        self.assertIsNot(list(first_result)[0], list(second_result)[0])

    def test_undeclared_keys_are_not_added_to_the_table(self):
        decoder = json.JSONDecoder(reusable_keys=self.keys)
        before = dict(decoder.reusable_keys)
        for document in (
                u'{"device": 1, "other": 2}',
                u'{"other": 3, "temp": 4}',
                u'{"device": {"other": 5}}'):
            decoder.decode(document)
        self.assertEqual(decoder.reusable_keys, before)
        self.assertEqual(set(decoder.reusable_keys),
                         set([self.device, self.temp]))
        self.assertNotIn(self.other, decoder.reusable_keys)

    def test_failed_document_does_not_leak_context(self):
        decoder = json.JSONDecoder(reusable_keys=self.keys)
        decoder.decode(u'{"device": 1, "other": 2}')
        with self.assertRaises(json.JSONDecodeError):
            decoder.decode(u'{"device": 3, "other":')
        self.assertEqual(decoder.memo, {})
        result = decoder.decode(u'{"device": 4, "other": 5}')
        self.assertIs(list(result)[0], self.device)
        self.assertIsNot(list(result)[1], self.device)
        self.assertEqual(set(decoder.reusable_keys),
                         set([self.device, self.temp]))

    def test_loads_accepts_reusable_keys(self):
        result = json.loads(
            u'{"device": {"device": 1}}', reusable_keys=self.keys)
        self.assertIs(list(result)[0], self.device)
        self.assertIs(list(result[self.device])[0], self.device)

    def test_default_decoder_does_not_share_declared_objects(self):
        decoder = json.JSONDecoder()
        result = decoder.decode(u'{"device": 1}')
        self.assertEqual(result, {'device': 1})
        self.assertIsNot(list(result)[0], self.device)

    def test_reusable_keys_must_be_strings(self):
        self.assertRaises(TypeError, json.JSONDecoder,
                          reusable_keys=[self.device, 1])

    def test_reusable_keys_rejects_single_string(self):
        self.assertRaises(TypeError, json.JSONDecoder,
                          reusable_keys=self.device)
