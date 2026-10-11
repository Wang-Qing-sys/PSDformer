"""Data contracts: train-only statistics, absent samples, and leakage-safe windows."""
import csv
from pathlib import Path
import tempfile
import unittest

import numpy as np

from psdformer.data import StandardNormalizer, load_npz, prepare_csv, split_subjects, validate_split, partition_indices


class NormalizationTests(unittest.TestCase):
    def test_fit_uses_only_observed_values(self):
        train = np.array([[[1.0, 10.0], [3.0, 20.0]], [[5.0, 30.0], [7.0, 999.0]]], dtype=np.float32)
        mask = np.ones_like(train, dtype=bool)
        mask[1, 1, 1] = False
        normalizer = StandardNormalizer.fit(train, mask)
        transformed = normalizer.transform(train, mask)
        expected = np.stack(((train[..., 0] - 4.0) / np.sqrt(5.0),
                             (train[..., 1] - 20.0) / np.sqrt(200.0 / 3.0)), axis=-1)
        expected[~mask] = 0
        np.testing.assert_allclose(transformed, expected, atol=1e-6, rtol=1e-6)
        restored = normalizer.inverse_transform(transformed)
        np.testing.assert_allclose(restored[mask], train[mask], atol=2e-5, rtol=1e-6)
        roundtrip = StandardNormalizer.from_dict(normalizer.to_dict())
        np.testing.assert_array_equal(roundtrip.transform(train, mask), transformed)

    def test_transform_does_not_refit_on_shifted_holdout(self):
        train = np.array([[[1.0], [3.0]], [[5.0], [7.0]]], dtype=np.float32)
        normalizer = StandardNormalizer.fit(train)
        before = normalizer.to_dict()
        held_out = np.full((1, 2, 1), 1e6, dtype=np.float32)
        transformed = normalizer.transform(held_out)
        self.assertGreater(float(transformed.min()), 1e4)
        self.assertEqual(normalizer.to_dict(), before)

    def test_nonfinite_masked_and_constant_inputs_are_finite(self):
        values = np.array([[[2.0, float('nan')], [2.0, 7.0], [2.0, float('inf')]]], dtype=np.float32)
        observed = np.isfinite(values)
        normalizer = StandardNormalizer.fit(values, observed)
        transformed = normalizer.transform(values, observed)
        self.assertTrue(np.isfinite(transformed).all())
        np.testing.assert_array_equal(transformed, np.zeros_like(values))


    def test_fully_missing_training_variable_fails_explicitly(self):
        with self.assertRaises(ValueError):
            StandardNormalizer.fit(np.full((2, 3, 1), np.nan))

    def test_target_transform_preserves_missing_targets(self):
        values = np.array([[[1.0], [3.0]], [[5.0], [np.nan]]], dtype=np.float32)
        normalizer = StandardNormalizer.fit(values)
        actual = normalizer.transform(values)
        np.testing.assert_array_equal(np.isnan(actual), np.isnan(values))


class SubjectSplitTests(unittest.TestCase):
    def test_subjects_are_disjoint_exhaustive_and_reproducible(self):
        ids = np.repeat(['a', 'b', 'c', 'd', 'e', 'f'], [3, 1, 7, 2, 4, 2])
        split = split_subjects(ids, seed=71)
        self.assertEqual(split_subjects(ids, seed=71), split)
        self.assertEqual(validate_split(split, ids), split)
        groups = [set(split[p]) for p in ('train', 'val', 'test')]
        self.assertEqual(set.union(*groups), set(ids))
        self.assertFalse(groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
        indices = [partition_indices(ids, split, p) for p in ('train', 'val', 'test')]
        self.assertEqual(sorted(np.concatenate(indices).tolist()), list(range(len(ids))))
        for partition in ('train', 'val', 'test'):
            selected = partition_indices(ids, split, partition)
            self.assertEqual(set(ids[selected]), set(split[partition]))

    def test_overlap_unknown_missing_and_empty_partitions_are_rejected(self):
        ids = np.array(['a', 'b', 'c'])
        for split in (
            {'train': ['a'], 'val': ['a', 'b'], 'test': ['c']},
            {'train': ['a'], 'val': ['b'], 'test': ['d']},
            {'train': ['a'], 'val': [], 'test': ['b', 'c']},
            {'train': ['a', 'a'], 'val': ['b'], 'test': ['c']},
        ):
            with self.subTest(split=split), self.assertRaises(ValueError):
                validate_split(split, ids)
        with self.assertRaises(ValueError):
            split_subjects(np.array(['a', 'a', 'b']))


class CSVTests(unittest.TestCase):
    def test_future_alignment_and_subject_session_boundaries(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / 'measurements.csv'
            destination = Path(temporary) / 'windows.npz'
            rows = []
            for subject, offset in [('alice', 0), ('bob', 100)]:
                for session, shift in [('morning', 0), ('evening', 30)]:
                    for time in range(8):
                        rows.append(dict(subject=subject, session=session, time=time,
                                         feature=offset + shift + time,
                                         target=1000 + offset + shift + time,
                                         label=time % 3))
            # Deliberately shuffled CSV rows must be regrouped and ordered before windowing.
            np.random.default_rng(7).shuffle(rows)
            with source.open('w', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            prepare_csv(source, destination, feature_columns=['feature'],
                        target_columns=['target'], subject_column='subject',
                        session_column='session', time_column='time',
                        input_len=3, pred_len=2, stride=2, label_column='label', num_classes=3)
            data = load_npz(destination, input_len=3, input_dim=1, pred_len=2,
                            output_dim=1, num_classes=3)
            self.assertEqual(data['x'].shape, (8, 3, 1))
            self.assertEqual(data['target'].shape, (8, 2, 1))
            self.assertEqual(set(data['subject_ids'].tolist()), {'alice', 'bob'})
            for x, y, label, subject in zip(data['x'], data['target'], data['y'], data['subject_ids']):
                np.testing.assert_array_equal(np.diff(x[:, 0]), [1, 1])
                np.testing.assert_array_equal(y[:, 0], x[-1, 0] + np.array([1001, 1002]))
                self.assertEqual(int(label), int(x[-1, 0] % 10) % 3)
                self.assertEqual(subject, 'bob' if x[0, 0] >= 100 else 'alice')

    def test_npz_defaults_and_nonfinite_input_mask_are_explicit(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / 'unlabelled.npz'
            x = np.array([[[np.nan, 2.], [np.inf, 4.]]], dtype=np.float32)
            np.savez(source, x=x, subject_ids=np.array(['a']))
            data = load_npz(source, pred_len=3, output_dim=2)
            self.assertEqual(data['target'].shape, (1, 3, 2))
            self.assertTrue(np.isnan(data['target']).all())
            np.testing.assert_array_equal(data['y'], [-1])
            np.testing.assert_array_equal(data['mask'], np.isfinite(x))
            self.assertTrue(np.isfinite(data['x']).all())
            self.assertTrue((data['x'][~data['mask']] == 0).all())

    def test_npz_rejects_fractional_out_of_range_labels_and_bad_mask(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / 'invalid.npz'
            base = {'x': np.ones((1, 3, 2)), 'subject_ids': np.array(['a'])}
            for extra in ({'y': np.array([.5])}, {'y': np.array([-2])},
                          {'y': np.array([3])}, {'mask': np.full((1, 3, 2), 2)}):
                np.savez(source, **base, **extra)
                with self.subTest(extra=list(extra)), self.assertRaises(ValueError):
                    load_npz(source, num_classes=3)

    def test_classification_csv_needs_no_future_rows(self):
        with tempfile.TemporaryDirectory() as temporary:
            source, output = Path(temporary) / 'source.csv', Path(temporary) / 'out.npz'
            source.write_text('subject,session,time,feature,label\na,s,0,1,0\na,s,1,2,1\na,s,2,3,2\n')
            prepare_csv(source, output, ['feature'], [], 'subject', 'session', 'time',
                        input_len=3, pred_len=7, label_column='label', num_classes=3)
            data = load_npz(output, input_len=3, input_dim=1, pred_len=7, output_dim=1, num_classes=3)
            self.assertEqual(data['x'].shape, (1, 3, 1))
            self.assertTrue(np.isnan(data['target']).all())
            np.testing.assert_array_equal(data['y'], [2])

    def test_csv_duplicate_timestamp_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            source, output = Path(temporary) / 'source.csv', Path(temporary) / 'out.npz'
            source.write_text('subject,session,time,feature,target\na,s,0,1,2\na,s,0,2,3\na,s,1,3,4\n')
            with self.assertRaises(ValueError):
                prepare_csv(source, output, ['feature'], ['target'], 'subject', 'session', 'time',
                            input_len=1, pred_len=1)

    def test_npz_rejects_inconsistent_window_counts(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / 'invalid.npz'
            np.savez(source, x=np.ones((4, 3, 2)), target=np.ones((3, 2, 1)),
                     subject_ids=np.array(['a', 'b', 'c', 'd']))
            with self.assertRaises(ValueError):
                load_npz(source, input_len=3, input_dim=2, pred_len=2, output_dim=1)


if __name__ == '__main__':
    unittest.main()
