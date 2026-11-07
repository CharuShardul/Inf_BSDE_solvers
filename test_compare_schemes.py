"""Numerical prerequisites and data-selection checks for the comparison.

Run with the project dependencies: python -m unittest test_compare_schemes
"""

import unittest
import hashlib
import json
import tempfile
from pathlib import Path
import numpy as np

from compare_schemes import (configure_cpu, relative_errors, choose_config,
                             worker_signature, verify_reference, REFERENCE_DIR,
                             plan_additional_runs)

configure_cpu()


class ComparisonChecks(unittest.TestCase):
    def test_reference_data_checksums_and_run_counts(self):
        self.assertGreater(verify_reference(REFERENCE_DIR), 10)
        rows = json.loads((REFERENCE_DIR/'results.json').read_text())
        self.assertEqual(len(rows),120)
        self.assertEqual(sum(r['stage']=='tune' for r in rows),45)
        self.assertEqual(sum(r['stage']=='test' for r in rows),75)
        selected = json.loads((REFERENCE_DIR/'selection.json').read_text())
        for method in ['grid','picard','direct']:
            for budget in [5.,10.,20.,30.,100.]:
                final = [r for r in rows if r['stage']=='test'
                         and r['method']==method and r['budget']==budget]
                self.assertEqual(len(final),5)
                self.assertEqual({r['seed'] for r in final},set(range(2100,2105)))
                self.assertEqual({r['config_index'] for r in final},
                                 {selected[method][str(budget)]['index']})
                self.assertTrue(all(r['solve_seconds']<=budget for r in final))

    def test_repeat_extension_preserves_selection_and_uses_only_fresh_final_seeds(self):
        rows = json.loads((REFERENCE_DIR/'results.json').read_text())
        original = [r for r in rows if r['stage']=='tune' or r['seed']<2103]
        selected = json.loads((REFERENCE_DIR/'selection.json').read_text())
        budgets = [5.,10.,20.,30.,100.]
        jobs = plan_additional_runs(original,selected,budgets,5)
        self.assertEqual(len(jobs),30)
        self.assertEqual({job['stage'] for job in jobs},{'test'})
        self.assertEqual({job['seed'] for job in jobs},{2103,2104})
        for job in jobs:
            self.assertEqual(job['config_index'],
                             selected[job['method']][str(job['budget'])]['index'])
        self.assertEqual(plan_additional_runs(original+jobs,selected,budgets,5),[])

    def test_repeat_extension_rejects_duplicate_seeds_or_changed_parameters(self):
        rows = json.loads((REFERENCE_DIR/'results.json').read_text())
        selected = json.loads((REFERENCE_DIR/'selection.json').read_text())
        final = next(r for r in rows if r['stage']=='test')
        with self.assertRaisesRegex(ValueError,'duplicated'):
            plan_additional_runs(rows+[final],selected,[5.,10.,20.,30.,100.],5)
        changed = dict(final,config_index=(final['config_index']+1)%3)
        with self.assertRaisesRegex(ValueError,'frozen selection'):
            plan_additional_runs(rows+[changed],selected,[5.,10.,20.,30.,100.],5)

    def test_reference_verification_detects_modified_data(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            (root/'results.json').write_text('[]')
            digest=hashlib.sha256((root/'results.json').read_bytes()).hexdigest()
            (root/'checksums.json').write_text(json.dumps({'results.json':digest}))
            self.assertEqual(verify_reference(root),1)
            (root/'results.json').write_text('[1]')
            with self.assertRaises(ValueError):
                verify_reference(root)

    def test_cache_reuse_ignores_plot_edits_but_rejects_numerical_changes(self):
        original = 'CONFIGS = {"grid": [1]}\ndef run_trial(job):\n    return job["budget"]\ndef plot_results():\n    return "old title"\n'
        restyled = original.replace('old title','new title')
        changed_worker = original.replace('return job["budget"]','return 2*job["budget"]')
        self.assertEqual(worker_signature(original),worker_signature(restyled))
        self.assertNotEqual(worker_signature(original),worker_signature(changed_worker))

    def test_interpolation_preserves_samples_and_clips_only_coordinates(self):
        from vect_grid_scheme import GridScheme
        grid = GridScheme(Ntilde=10, R=6, r=0, M=1, plot_iter=False)
        points = np.array([[[-9.], [-.37], [0.]], [[.73], [6.], [10.]]])
        original = points.copy()
        prediction = grid.interpol(points, grid.x_grid)
        np.testing.assert_array_equal(points, original)
        np.testing.assert_allclose(prediction, np.clip(original, -6., 6.), atol=1e-14)

    def test_multidimensional_interpolation(self):
        from vect_grid_scheme import GridScheme
        grid = GridScheme(d=2, Ntilde=4, R=2, r=0, M=1, plot_iter=False)
        points = np.array([[-.23,.51], [2.5,-3.]])
        values = grid.x_grid[:, :1]+2*grid.x_grid[:, 1:]
        clipped = np.clip(points,-2.,2.)
        np.testing.assert_allclose(grid.interpol(points,values),
                                   clipped[:, :1]+2*clipped[:, 1:], atol=1e-14)

    def test_relative_errors_use_the_vector_norm(self):
        points = np.random.default_rng(4).normal(size=(50, 2))
        truth = np.concatenate([np.mean(np.arctan(points),axis=-1,keepdims=True),
                                1/(2*(1+points**2))],axis=1)
        self.assertEqual(relative_errors(truth,points)["score"],0.)
        self.assertAlmostEqual(relative_errors(2*truth,points)["score"],1.)

    def test_test_errors_do_not_select_parameters(self):
        rows = [dict(stage="tune",method="grid",budget=10.,config_index=i,
                     validation={"score":v}) for i,v in enumerate([.3,.1,.2])]
        rows.append(dict(stage="test",method="grid",budget=10.,config_index=0,
                         test={"score":0.}))
        self.assertEqual(choose_config(rows,"grid",10.)[0],1)

    def test_neural_labels_precision_generator_and_training(self):
        import tensorflow as tf
        from NN_Picard_mult_alt import NNPicardSolver
        from NN_direct_scheme import NNDirecSolver
        from vect_grid_scheme import GridScheme
        tf.config.set_visible_devices([],"GPU")
        tf.keras.backend.set_floatx("float64")
        tf.keras.utils.set_random_seed(52)
        picard = NNPicardSolver(M=16,batch_size=8,epochs=1,K_z=.1)
        direct = NNDirecSolver(M=4,batch_size=8,n_steps=10,n_decays=2,K_z=.1)
        self.assertEqual(picard.NN.count_params(),direct.NN.count_params())
        self.assertEqual(picard.NN.count_params(),716)
        grid = GridScheme(M=4,K_z=.1,plot_iter=False)
        points = np.random.normal(size=(16,1))
        y = np.random.normal(size=(16,1))
        z = np.random.normal(size=(16,1))
        np.testing.assert_allclose(picard.f(points,y,z), grid.f(points,y,z),atol=1e-14)
        np.testing.assert_allclose(direct.f(points,y,z), grid.f(points,y,z),atol=1e-14)
        labels = picard.label(points,picard.model_init)
        self.assertEqual(labels.dtype,tf.float64)
        self.assertTrue(np.isfinite(labels).all())
        before = picard.NN.get_weights()
        picard.NN.train_on_batch(points,labels)
        self.assertTrue(any(not np.array_equal(a,b) for a,b in zip(before,picard.NN.get_weights())))
        labels = direct.label(points,direct.model,training=False)
        self.assertEqual(labels.dtype,tf.float64)
        self.assertTrue(np.isfinite(labels).all())
        before = direct.NN.get_weights()
        direct.model.train_step(points,labels)
        self.assertEqual(int(direct.model.optimizer.iterations.numpy()),1)
        self.assertTrue(any(not np.array_equal(a,b) for a,b in zip(before,direct.NN.get_weights())))


if __name__ == "__main__":
    unittest.main()
