import json
from pathlib import Path
import re
import tempfile
import unittest
import zipfile

from kpt_test.adapters import load_xes3g5m
from kpt_test.bundle import build_da20k
from kpt_test.evaluate import evaluate_submission
from kpt_test.io import file_hash, index_records, read_json, read_records, write_json
from kpt_test.taxonomy import Taxonomy
from kpt_test.text import normalize_question_text


class TextPreparationTests(unittest.TestCase):
    def test_mathml_fractions_and_powers_are_preserved(self):
        text = '<div>计算<math><mfrac><mn>1</mn><mn>2</mn></mfrac><mo>+</mo><msup><mi>x</mi><mn>2</mn></msup></math></div>'
        self.assertEqual(normalize_question_text(text), r'计算$\frac{1}{2}+{x}^{2}$')

    def test_matrices_and_prescripts(self):
        matrix = '<math><mtable><mtr><mtd><mn>1</mn></mtd><mtd><mn>2</mn></mtd></mtr></mtable></math>'
        self.assertIn(r'\begin{matrix}1 & 2\end{matrix}', normalize_question_text(matrix))
        prescript = '<math><mmultiscripts><mi>C</mi><mprescripts/><none/><mn>14</mn></mmultiscripts></math>'
        self.assertIn('{}_{' + '}^{14}{C}', normalize_question_text(prescript))

    def test_negated_subset_is_not_changed_to_positive_subset(self):
        text = '<math><menclose notation="updiagonalstrike"><mo>⊂</mo></menclose></math>'
        self.assertEqual(normalize_question_text(text), r'$\not\subset$')

    def test_log_function_remains_distinct_from_multiplication(self):
        self.assertEqual(normalize_question_text('<math><mi>ln</mi><mi>x</mi></math>'), r'$\ln x$')

    def test_images_footer_and_scripts_do_not_enter_input(self):
        text = '<div>题目<img src="https://example.com/a.png"><script>secret</script></div><div class="exam-foot"><a>查看解析 secret</a></div>'
        self.assertEqual(normalize_question_text(text), '题目')
        self.assertEqual(normalize_question_text('计算 $$1+1$$\nquestion_1-image_0'), '计算 $$1+1$$')

    def test_plain_inequalities_survive(self):
        self.assertEqual(normalize_question_text('若 x<y 且 z>1，求值。'), '若 x<y 且 z>1，求值。')


class RealFormatTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_xes_node_names_multi_parent_and_route_endpoints(self):
        nodes = {'0': '根甲', '1': '根乙', '2': '父节点', '3': '知识点', '4': ' 知识点', '5': '后代'}
        questions = {
            'q1': {'content': '题目一', 'kc_routes': ['根甲----父节点----知识点']},
            'q2': {'content': '题目二', 'kc_routes': ['根乙----父节点----知识点']},
            'q3': {'content': '题目三', 'kc_routes': ['根甲----父节点---- 知识点']},
            'q4': {'content': '题目四', 'kc_routes': ['根甲----父节点----知识点----后代']},
        }
        write_json(self.root / 'questions.json', questions)
        write_json(self.root / 'nodes.json', nodes)
        rows, taxonomy, info = load_xes3g5m(self.root / 'questions.json', self.root / 'nodes.json')
        self.assertEqual(taxonomy.ids, ['3', '4', '5'])
        self.assertEqual(len(taxonomy.routes('3')), 2)
        self.assertEqual(rows[2]['labels'], ['4'])
        self.assertEqual(taxonomy.hierarchy_score('3', '4'), .5)
        self.assertIn('根乙', taxonomy.text('3'))
        self.assertEqual(Taxonomy(taxonomy.data).digest, taxonomy.digest)
        self.assertEqual(info['source_nodes'], 6)

    def test_da_archive_filtering_is_recorded(self):
        questions = [{'id': i, 'text': '<div>计算 x+1。</div>'} for i in range(14)]
        questions[3]['text'] = '<div>3、<img src="a.png"></div>'
        links = [{'qid': i, 'label_id': 3} for i in range(1, 14) if i != 2]
        links.extend([{'qid': 2, 'label_id': 99}, {'qid': 4, 'label_id': 99}])
        archive = self.root / 'mathdata.zip'
        with zipfile.ZipFile(archive, 'w') as z:
            z.writestr('mathdata-main/math_questions_content.json', json.dumps(questions))
            z.writestr('mathdata-main/math_questions_knowledgetag.json', json.dumps(links))
        labels = self.root / 'labels.json'
        write_json(labels, [{'label_id': 3, 'name': '加法', 'label_father_id': 'f0', 'label_father_name': '运算'}])
        result = build_da20k(archive, labels, self.root / 'output')
        self.assertEqual(result['adapter']['included_questions'], 11)
        self.assertEqual(result['adapter']['exclusion_counts'], {'no_tags': 1, 'no_target_labels': 1, 'no_text': 1})
        audit = read_json(self.root / 'output/provenance.json')
        self.assertEqual(audit['filtered_tags'], [{'qid': '4', 'ignored_labels': ['99'], 'kept_labels': ['3']}])


class BundledDatasetTests(unittest.TestCase):
    """Verify the real, checked-in data without network or model downloads."""

    def test_real_xes_source_produces_7652_questions_and_865_targets(self):
        root = Path(__file__).resolve().parents[1] / 'datasets'
        rows, taxonomy, _ = load_xes3g5m(root / 'sources/xes3g5m/questions.json', root / 'sources/xes3g5m/kc_routes_map.json')
        self.assertEqual(len(rows), 7652)
        self.assertEqual(len(taxonomy.ids), 865)
        self.assertEqual(taxonomy.digest, Taxonomy(read_json(root / 'xes3g5m/taxonomy.json')).digest)

    def test_bundled_files_are_complete_text_only_and_disjoint(self):
        root = Path(__file__).resolve().parents[1] / 'datasets'
        for dataset in ['da20k', 'xes3g5m']:
            with self.subTest(dataset=dataset):
                bundle = root / dataset
                manifest = read_json(bundle / 'manifest.json')
                for filename, digest in manifest['files'].items():
                    self.assertEqual(file_hash(bundle / filename), digest, filename)
                splits = read_json(bundle / 'splits.json')
                all_ids = [qid for ids in splits.values() for qid in ids]
                self.assertEqual(len(all_ids), len(set(all_ids)))
                self.assertEqual(len(all_ids), manifest['adapter']['included_questions'])
                for filename in ['train.jsonl', 'valid.jsonl', 'test.inputs.jsonl', 'test.noisy.inputs.jsonl']:
                    for row in read_records(bundle / filename):
                        self.assertTrue(row['text'].strip())
                        self.assertNotRegex(row['text'], r'<img\b|\bquestion_\d+-image_\d+\b|https?://')
                gold = index_records(read_records(bundle / 'test.gold.jsonl'), 'labels')
                self.assertEqual(set(gold), set(splits['test']))

    def test_offline_evaluation_on_real_gold_as_oracle(self):
        # Gold-as-prediction only checks evaluator correctness, not model quality.
        root = Path(__file__).resolve().parents[1] / 'datasets'
        for dataset in ['da20k', 'xes3g5m']:
            with self.subTest(dataset=dataset):
                bundle = root / dataset
                gold = bundle / 'test.gold.jsonl'
                report, _ = evaluate_submission(bundle, gold, gold)
                self.assertTrue(report['evaluation_complete'])
                self.assertEqual(report['metrics']['strict_accuracy'], 1)
                self.assertEqual(report['metrics']['exact']['micro_f1'], 1)
                self.assertEqual(report['metrics']['semantic']['f1'], 1)
                self.assertEqual(report['interaction_robustness']['ratio'], 1)


if __name__ == '__main__':
    unittest.main()
