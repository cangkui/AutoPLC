import json
from tqdm import tqdm
import os
from collections import defaultdict, Counter

def eval_api_recommendation(task_folder: str,
                            ground_truth_path: str,
                            allowed_actual: list = []):
    
    # Validate ground_truth_path
    if not os.path.exists(ground_truth_path):
        raise FileNotFoundError(f"Ground truth file not found: {ground_truth_path}")

    # Validate task_folder
    if not os.path.exists(task_folder):
        raise FileNotFoundError(f"Task folder not found: {task_folder}")

    with open(ground_truth_path, 'r', encoding='utf-8') as f:
        gt_data = json.load(f)
    gt_data = {k.lower(): v for k, v in gt_data.items()}

    # IF allowed_actual is provided, filter the ground truth APIs
    allowed_set = set(func.lower() for func in allowed_actual)

    total_tp, total_fp, total_fn = 0, 0, 0
    task_names = [name for name in os.listdir(task_folder)
                  if os.path.isdir(os.path.join(task_folder, name))]
    max_gt_len = 0
    max_pred_len = 0

    for task in tqdm(task_names, desc="Evaluating API Recommendations"):
        inter_path = os.path.join(task_folder, task, "intermediate_results.json")
        if not os.path.exists(inter_path):
            continue

        with open(inter_path, 'r', encoding='utf-8') as f:
            inter_data = json.load(f)

        # filter APIs in the intermediate data
        predicted = set(api.lower()
                        for api in inter_data.get("apis_for_this_task", [])
                        if "_TO_" not in api)
        actual = set(api.lower()
                     for api in gt_data.get(task.lower(), [])
                     if "_TO_" not in api)

        # Filter by allowed set if provided
        if allowed_set:
            actual = actual.intersection(allowed_set)
            predicted = predicted.intersection(allowed_set)

        # Compute stats for this case
        tp = len(predicted & actual)
        fp_items = sorted(predicted - actual)
        fn_items = sorted(actual - predicted)
        fp = len(fp_items)
        fn = len(fn_items)

        # Accumulate totals
        total_tp += tp
        total_fp += fp
        total_fn += fn

        # Track max sizes
        max_gt_len = max(max_gt_len, len(actual))
        max_pred_len = max(max_pred_len, len(predicted))

        # Print per-case missing and wrong recommendations
        print(f"\nCase: {task}")
        if fn_items:
            print(f"  Missing (FN={fn}): {fn_items}")
        else:
            print("  Missing (FN=0): None")
        if fp_items:
            print(f"  Wrong recommendations (FP={fp}): {fp_items}")
        else:
            print("  Wrong recommendations (FP=0): None")
            
            
    # Compute overall precision, recall, f1
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

    print("\n=== API Recommendation Evaluation Summary ===")
    print(f"Total TP: {total_tp}, FP: {total_fp}, FN: {total_fn}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"Max GT Length: {max_gt_len}")
    print(f"Max Predicted Length: {max_pred_len}")


import re

def check_api_in_code(api_name: str, code: str) -> bool:
    """
    检查 api_name 是否作为独立标识符出现在 code 中。
    要求其前后不能是字母或数字。
    """
    # \b 表示单词边界，防止 "myprint" 匹配 "print"
    # re.IGNORECASE 实现不区分大小写匹配
    pattern = r'\b' + re.escape(api_name) + r'\b'
    return re.search(pattern, code, re.IGNORECASE) is not None

def eval_api_recommendation_advanced(task_folder: str,
                            ground_truth_path: str,
                            allowed_actual: list = [],
                            api_category_path: str = None,
                            api_complexity_path: str = None):
    
    # Validate inputs
    if not os.path.exists(ground_truth_path):
        raise FileNotFoundError(f"Ground truth file not found: {ground_truth_path}")
    if not os.path.exists(task_folder):
        raise FileNotFoundError(f"Task folder not found: {task_folder}")

    with open(ground_truth_path, 'r', encoding='utf-8') as f:
        gt_data = json.load(f)
    gt_data = {k.lower(): v for k, v in gt_data.items()}

    # Load API category mapping if provided
    api_categories = {}
    if api_category_path and os.path.exists(api_category_path):
        with open(api_category_path, 'r', encoding='utf-8') as f:
            api_categories = json.load(f)
        api_categories = {k.lower(): v for k, v in api_categories.items()}

    # Load API complexity mapping if provided
    api_complexity = {}
    if api_complexity_path and os.path.exists(api_complexity_path):
        with open(api_complexity_path, 'r', encoding='utf-8') as f:
            api_complexity = json.load(f)
        api_complexity = {k.lower(): v for k, v in api_complexity.items()}

    allowed_set = set(func.lower() for func in allowed_actual)

    # Initialize metrics
    total_tp, total_fp, total_fn = 0, 0, 0
    task_results = []
    category_stats = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})
    complexity_stats = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})
    
    # API frequency analysis
    gt_api_freq = Counter()
    pred_api_freq = Counter()
    
    # Critical API tracking (APIs that appear in >50% of tasks)
    all_gt_apis = set()
    critical_apis_performance = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0, 'total_tasks': 0})
    
    # Ranking metrics
    ranking_scores = []
    coverage_at_k = {k: [] for k in [1, 3, 5, 10]}
    
    task_names = [name for name in os.listdir(task_folder)
                  if os.path.isdir(os.path.join(task_folder, name))]

    # Code usage analysis metrics
    code_usage_stats = defaultdict(lambda: {'predicted_used': 0, 'predicted_unused': 0, 'gt_used': 0, 'gt_unused': 0})
    api_usage_freq = Counter()
    predicted_but_unused = Counter()
    llm_knowledge_apis = Counter()  # APIs used but not recommended (LLM inherent knowledge)
    
    for task in tqdm(task_names, desc="Evaluating API Recommendations"):
        inter_path = os.path.join(task_folder, task, "intermediate_results.json")
        
        # 读取task_folder, task, 下面的*.st或者*.scl作为代码文件 (肯定只有一个)
        code_content = ""
        task_path = os.path.join(task_folder, task)
        if os.path.exists(task_path):
            # Look for .st or .scl files
            code_files = [f for f in os.listdir(task_path) if f.endswith(('.st', '.scl'))]
            if code_files:
                code_file_path = os.path.join(task_path, code_files[0])
                try:
                    with open(code_file_path, 'r', encoding='utf-8') as cf:
                        code_content = cf.read().lower()  # Convert to lowercase for case-insensitive matching
                except Exception as e:
                    print(f"Warning: Could not read code file {code_file_path}: {e}")
                    raise FileNotFoundError(f"Code file not found or unreadable: {code_file_path}")
                
        if not os.path.exists(inter_path):
            continue

        with open(inter_path, 'r', encoding='utf-8') as f:
            inter_data = json.load(f)

        # Get predicted APIs with ranking if available
        predicted_list = [api.lower() for api in inter_data.get("apis_for_this_task", []) if "_TO_" not in api]
        predicted = set(api for api in predicted_list)
        
        actual = set(api.lower() for api in gt_data.get(task.lower(), []) if "_TO_" not in api)

        # 去掉'not'，this is not a true API
        actual = {api for api in actual if not api.startswith('not')}

        # Filter by allowed set if provided
        if allowed_set:
            actual = actual.intersection(allowed_set)
            predicted = predicted.intersection(allowed_set)
            predicted_list = [p for p in predicted_list if p in allowed_set]
        
        # Code usage analysis - check which APIs actually appear in the generated code
        predicted_used_in_code = set()
        predicted_unused_in_code = set()
        actual_used_in_code = set()
        actual_unused_in_code = set()
        used_but_not_recommended = set()  # LLM inherent knowledge APIs
        
        if code_content:
            # Check predicted APIs usage in code
            for api in predicted:
                # Check if API name appears in the code (case-insensitive)
                if check_api_in_code(api, code_content):
                    predicted_used_in_code.add(api)
                    api_usage_freq[api] += 1
                else:
                    predicted_unused_in_code.add(api)
                    predicted_but_unused[api] += 1
            
            # Check actual APIs usage in code
            for api in actual:
                if check_api_in_code(api, code_content):
                    actual_used_in_code.add(api)
                else:
                    actual_unused_in_code.add(api)
            
            # Find all APIs used in code (simple pattern matching)
            used_apis_in_code = set()
            for api_ in allowed_actual:
                if check_api_in_code(api_, code_content):
                    used_apis_in_code.add(api_.lower())
            
            # Filter to only include known APIs (from ground truth or predictions)
            all_known_apis = set(gt_api_freq.keys()) | set(pred_api_freq.keys())
            used_apis_in_code = used_apis_in_code & all_known_apis
            
            # Find APIs used but not recommended (LLM inherent knowledge)
            used_but_not_recommended = used_apis_in_code - predicted
            for api in used_but_not_recommended:
                llm_knowledge_apis[api] += 1
        
        # Update code usage statistics
        code_usage_stats['all']['predicted_used'] += len(predicted_used_in_code)
        code_usage_stats['all']['predicted_unused'] += len(predicted_unused_in_code)
        code_usage_stats['all']['gt_used'] += len(actual_used_in_code)
        code_usage_stats['all']['gt_unused'] += len(actual_unused_in_code)
        
        # Update API frequency
        for api in actual:
            gt_api_freq[api] += 1
            all_gt_apis.add(api)
        for api in predicted:
            pred_api_freq[api] += 1

        # Basic metrics
        tp = len(predicted & actual)
        fp_items = sorted(predicted - actual)
        fn_items = sorted(actual - predicted)
        fp = len(fp_items)
        fn = len(fn_items)

        # Handle edge cases properly
        if len(actual) == 0:  # Ground truth is empty
            if len(predicted) == 0:
                # Both empty - perfect match
                task_precision = 1.0
                task_recall = 1.0  # Convention: 1.0 when both are empty
            else:
                # GT empty but has predictions - all are false positives
                task_precision = 0.0
                task_recall = 1.0  # No true positives to miss
        else:  # Ground truth is not empty
            task_precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
            task_recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0

        # Store task result
        task_result = {
            'task': task,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'precision': task_precision,
            'recall': task_recall,
            'f1': 2 * task_precision * task_recall / (task_precision + task_recall) if (task_precision + task_recall) > 0 else 0,
            'gt_empty': len(actual) == 0,
            'pred_empty': len(predicted) == 0,
            'actual_apis': list(actual),
            'predicted_apis': list(predicted),
            'fp_items': fp_items,
            'fn_items': fn_items,
            'predicted_used_in_code': list(predicted_used_in_code),
            'predicted_unused_in_code': list(predicted_unused_in_code),
            'actual_used_in_code': list(actual_used_in_code),
            'actual_unused_in_code': list(actual_unused_in_code),
            'used_but_not_recommended': list(used_but_not_recommended),
            'code_usage_rate': len(predicted_used_in_code) / len(predicted) if len(predicted) > 0 else 0,
            'llm_knowledge_count': len(used_but_not_recommended)
        }
        task_results.append(task_result)

        # Category-wise analysis
        for api in actual:
            category = api_categories.get(api, 'default')
            critical_apis_performance[api]['total_tasks'] += 1
            
        for api in predicted & actual:  # TP
            category = api_categories.get(api, 'default')
            category_stats[category]['tp'] += 1
            critical_apis_performance[api]['tp'] += 1
            
        for api in predicted - actual:  # FP
            category = api_categories.get(api, 'default')
            category_stats[category]['fp'] += 1
            critical_apis_performance[api]['fp'] += 1
            
        for api in actual - predicted:  # FN
            category = api_categories.get(api, 'default')
            category_stats[category]['fn'] += 1
            critical_apis_performance[api]['fn'] += 1

        # Complexity-wise analysis
        for api in predicted & actual:  # TP
            complexity = api_complexity.get(api, 'default')
            complexity_stats[complexity]['tp'] += 1
            
        for api in predicted - actual:  # FP
            complexity = api_complexity.get(api, 'default')
            complexity_stats[complexity]['fp'] += 1
            
        for api in actual - predicted:  # FN
            complexity = api_complexity.get(api, 'default')
            complexity_stats[complexity]['fn'] += 1

        # Ranking metrics (if prediction order is available)
        if len(predicted_list) > 0 and len(actual) > 0:
            # Calculate MAP (Mean Average Precision)
            relevant_positions = []
            for i, pred_api in enumerate(predicted_list):
                if pred_api in actual:
                    relevant_positions.append(i + 1)
            
            if relevant_positions:
                avg_precision = sum((j + 1) / pos for j, pos in enumerate(relevant_positions)) / len(actual)
                ranking_scores.append(avg_precision)
            else:
                ranking_scores.append(0.0)
            
            # Coverage@K
            for k in coverage_at_k.keys():
                if k <= len(predicted_list):
                    coverage = len(set(predicted_list[:k]) & actual) / len(actual) if len(actual) > 0 else 0
                    coverage_at_k[k].append(coverage)

        # Or use weighted approach
        if len(actual) > 0:
            total_tp += tp
            total_fp += fp
            total_fn += fn
        else:
            # Handle empty GT cases separately
            if len(predicted) == 0:
                # Perfect match for empty case
                pass  # Don't penalize
            else:
                # False predictions for empty GT
                total_fp += fp 

    # Identify critical APIs (appear in >50% of tasks)
    critical_apis = {api for api, freq in gt_api_freq.items() if freq > len(task_names) * 0.5} 

    non_empty_tasks = [r for r in task_results if not r['gt_empty']]
    if non_empty_tasks:
        overall_precision = sum(r['precision'] * (r['tp'] + r['fp']) for r in non_empty_tasks) / sum(r['tp'] + r['fp'] for r in non_empty_tasks) if sum(r['tp'] + r['fp'] for r in non_empty_tasks) > 0 else 0
        overall_recall = sum(r['recall'] * (r['tp'] + r['fn']) for r in non_empty_tasks) / sum(r['tp'] + r['fn'] for r in non_empty_tasks) if sum(r['tp'] + r['fn'] for r in non_empty_tasks) > 0 else 0
    else:
        overall_precision = 1.0  # All tasks have empty GT
        overall_recall = 1.0

    # Option 2: Include empty GT tasks with special handling
    macro_precision = sum(r['precision'] for r in task_results) / len(task_results) if task_results else 0
    macro_recall = sum(r['recall'] for r in task_results) / len(task_results) if task_results else 0


    # Print comprehensive results
    print("\n" + "="*60) 
    print("API RECOMMENDATION EVALUATION REPORT") 
    print("="*60) 

    # Edge case analysis
    empty_gt_tasks = sum(1 for r in task_results if r['gt_empty'])
    empty_pred_tasks = sum(1 for r in task_results if r['pred_empty'])
    perfect_empty_matches = sum(1 for r in task_results if r['gt_empty'] and r['pred_empty'])
    
    print(f"\n🔍 EDGE CASE ANALYSIS")
    print(f"Tasks with empty Ground Truth: {empty_gt_tasks}/{len(task_results)}")
    print(f"Tasks with empty Predictions: {empty_pred_tasks}/{len(task_results)}")
    print(f"Perfect empty matches: {perfect_empty_matches}/{empty_gt_tasks}")
    

    # Basic metrics with different calculation methods
    print(f"\n📊 OVERALL PERFORMANCE")
    print(f"Micro-averaged (excluding empty GT):")
    print(f"  Precision: {overall_precision:.4f}")
    print(f"  Recall:    {overall_recall:.4f}")
    print(f"  F1 Score:  {2 * overall_precision * overall_recall / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0:.4f}")
    
    print(f"Macro-averaged (including empty GT):")
    print(f"  Precision: {macro_precision:.4f}")
    print(f"  Recall:    {macro_recall:.4f}")
    print(f"  F1 Score:  {2 * macro_precision * macro_recall / (macro_precision + macro_recall) if (macro_precision + macro_recall) > 0 else 0:.4f}")
    
    # Ranking metrics
    if ranking_scores:
        map_score = sum(ranking_scores) / len(ranking_scores)
        print(f"Mean Average Precision (MAP): {map_score:.4f}")
        
        print(f"\n📈 COVERAGE@K METRICS")
        for k, scores in coverage_at_k.items():
            if scores:
                avg_coverage = sum(scores) / len(scores)
                print(f"Coverage@{k}: {avg_coverage:.4f}")

    # Task distribution analysis
    task_precisions = [r['precision'] for r in task_results if r['precision'] > 0]
    task_recalls = [r['recall'] for r in task_results if r['recall'] > 0]
    
    print(f"\n📋 TASK-LEVEL DISTRIBUTION")
    print(f"Tasks with perfect precision: {sum(1 for r in task_results if r['precision'] == 1.0)}/{len(task_results)}")
    print(f"Tasks with perfect recall: {sum(1 for r in task_results if r['recall'] == 1.0)}/{len(task_results)}")
    print(f"Tasks with zero precision: {sum(1 for r in task_results if r['precision'] == 0.0)}/{len(task_results)}")
    print(f"Tasks with zero recall: {sum(1 for r in task_results if r['recall'] == 0.0)}/{len(task_results)}")
    
    if task_precisions:
        print(f"Average task precision: {sum(task_precisions)/len(task_precisions):.4f}")
    if task_recalls:
        print(f"Average task recall: {sum(task_recalls)/len(task_recalls):.4f}")

    # Category-wise performance
    if category_stats:
        print(f"\n🏷️  CATEGORY-WISE PERFORMANCE")
        for category, stats in sorted(category_stats.items()):
            cat_precision = stats['tp'] / (stats['tp'] + stats['fp']) if (stats['tp'] + stats['fp']) > 0 else 0
            cat_recall = stats['tp'] / (stats['tp'] + stats['fn']) if (stats['tp'] + stats['fn']) > 0 else 0
            cat_f1 = 2 * cat_precision * cat_recall / (cat_precision + cat_recall) if (cat_precision + cat_recall) > 0 else 0
            print(f"{category:15} | P: {cat_precision:.3f} | R: {cat_recall:.3f} | F1: {cat_f1:.3f} | TP: {stats['tp']} | FP: {stats['fp']} | FN: {stats['fn']}")

    # Complexity-wise performance
    if complexity_stats:
        print(f"\n⚙️  COMPLEXITY-WISE PERFORMANCE")
        for complexity, stats in sorted(complexity_stats.items()):
            comp_precision = stats['tp'] / (stats['tp'] + stats['fp']) if (stats['tp'] + stats['fp']) > 0 else 0
            comp_recall = stats['tp'] / (stats['tp'] + stats['fn']) if (stats['tp'] + stats['fn']) > 0 else 0
            comp_f1 = 2 * comp_precision * comp_recall / (comp_precision + comp_recall) if (comp_precision + comp_recall) > 0 else 0
            print(f"{complexity:10} | P: {comp_precision:.3f} | R: {comp_recall:.3f} | F1: {comp_f1:.3f} | TP: {stats['tp']} | FP: {stats['fp']} | FN: {stats['fn']}")

    # Critical APIs performance
    if critical_apis:
        print(f"\n🎯 CRITICAL APIs PERFORMANCE (>50% frequency)")
        critical_performance = []
        for api in critical_apis:
            stats = critical_apis_performance[api]
            if stats['total_tasks'] > 0:
                api_recall = stats['tp'] / (stats['tp'] + stats['fn']) if (stats['tp'] + stats['fn']) > 0 else 0
                critical_performance.append((api, api_recall, stats))
        
        critical_performance.sort(key=lambda x: x[1])  # Sort by recall
        print("Worst performing critical APIs:")
        for api, recall, stats in critical_performance[:5]:
            print(f"  {api:20} | Recall: {recall:.3f} | Usage: {stats['total_tasks']}/{len(task_names)}")

    # API frequency analysis
    print(f"\n📊 API FREQUENCY ANALYSIS")
    print(f"Total unique APIs in ground truth: {len(gt_api_freq)}")
    print(f"Total unique APIs predicted: {len(pred_api_freq)}")
    
    # Most frequently missed APIs
    missed_freq = Counter()
    for task_result in task_results:
        for api in task_result['fn_items']:
            missed_freq[api] += 1
    
    if missed_freq:
        print(f"\nMost frequently missed APIs:")
        for api, count in missed_freq.most_common(10):
            total_appearance = gt_api_freq[api]
            miss_rate = count / total_appearance if total_appearance > 0 else 0
            print(f"  {api:25} | Missed: {count}/{total_appearance} ({miss_rate:.1%})")

    # Code usage analysis
    print(f"\n💻 CODE USAGE ANALYSIS")
    total_predicted = sum(len(r['predicted_apis']) for r in task_results)
    total_predicted_used = sum(len(r['predicted_used_in_code']) for r in task_results)
    total_predicted_unused = sum(len(r['predicted_unused_in_code']) for r in task_results)
    total_actual = sum(len(r['actual_apis']) for r in task_results)
    total_actual_used = sum(len(r['actual_used_in_code']) for r in task_results)
    
    if total_predicted > 0:
        usage_rate = total_predicted_used / total_predicted
        print(f"Predicted API usage rate: {usage_rate:.1%} ({total_predicted_used}/{total_predicted})")
    
    if total_actual > 0:
        gt_usage_rate = total_actual_used / total_actual
        print(f"Ground truth API usage rate: {gt_usage_rate:.1%} ({total_actual_used}/{total_actual})")
    
    # Task-level code usage distribution
    task_usage_rates = [r['code_usage_rate'] for r in task_results if len(r['predicted_apis']) > 0]
    if task_usage_rates:
        avg_task_usage = sum(task_usage_rates) / len(task_usage_rates)
        perfect_usage_tasks = sum(1 for rate in task_usage_rates if rate == 1.0)
        zero_usage_tasks = sum(1 for rate in task_usage_rates if rate == 0.0)
        
        print(f"Average task-level usage rate: {avg_task_usage:.1%}")
        print(f"Tasks with 100% API usage: {perfect_usage_tasks}/{len(task_usage_rates)}")
        print(f"Tasks with 0% API usage: {zero_usage_tasks}/{len(task_usage_rates)}")
    
    # Most frequently unused predicted APIs
    if predicted_but_unused:
        print(f"\nMost frequently predicted but unused APIs:")
        for api, count in predicted_but_unused.most_common(10):
            total_predicted_count = pred_api_freq[api]
            unused_rate = count / total_predicted_count if total_predicted_count > 0 else 0
            print(f"  {api:25} | Unused: {count}/{total_predicted_count} ({unused_rate:.1%})")
    
    # Precision considering code usage
    tp_and_used = sum(len(set(r['predicted_apis']) & set(r['actual_apis']) & set(r['predicted_used_in_code'])) for r in task_results)
    total_predicted_and_used = sum(len(r['predicted_used_in_code']) for r in task_results)
    
    if total_predicted_and_used > 0:
        precision_with_usage = tp_and_used / total_predicted_and_used
        print(f"\nPrecision among used APIs: {precision_with_usage:.4f}")
        print(f"  (Correct and used APIs / All used predicted APIs)")

    # LLM inherent knowledge analysis
    print(f"\n🧠 LLM INHERENT KNOWLEDGE ANALYSIS")
    total_llm_knowledge = sum(len(r['used_but_not_recommended']) for r in task_results)
    total_tasks_with_knowledge = sum(1 for r in task_results if len(r['used_but_not_recommended']) > 0)
    
    if total_llm_knowledge > 0:
        print(f"Total APIs used but not recommended: {total_llm_knowledge}")
        print(f"Tasks with LLM inherent knowledge usage: {total_tasks_with_knowledge}/{len(task_results)} ({total_tasks_with_knowledge/len(task_results):.1%})")
        
        avg_knowledge_per_task = sum(r['llm_knowledge_count'] for r in task_results) / len(task_results)
        print(f"Average LLM knowledge APIs per task: {avg_knowledge_per_task:.2f}")
        
        print(f"\nMost frequently used unrecommended APIs (LLM Knowledge):")
        for api, count in llm_knowledge_apis.most_common(20):
            # Check if this API appears in ground truth to see if it's actually useful
            gt_frequency = gt_api_freq.get(api, 0)
            if gt_frequency > 0:
                print(f"  {api:25} | Used: {count} times | GT freq: {gt_frequency}")
    else:
        print("No APIs used without recommendation detected.")

    # Industrial relevance metrics
    print(f"\n🏭 INDUSTRIAL RELEVANCE METRICS")
    
    # Safety-critical API coverage (if categories include safety info)
    safety_categories = ['safety', 'alarm', 'interlock', 'emergency']
    safety_apis = {api for api, cat in api_categories.items() if any(sc in cat.lower() for sc in safety_categories)}
    
    if safety_apis:
        safety_tp = sum(1 for result in task_results for api in result['actual_apis'] if api in safety_apis and api in result['predicted_apis'])
        safety_total = sum(1 for result in task_results for api in result['actual_apis'] if api in safety_apis)
        safety_coverage = safety_tp / safety_total if safety_total > 0 else 0
        print(f"Safety-critical API coverage: {safety_coverage:.1%} ({safety_tp}/{safety_total})")
    
    # Complexity distribution in recommendations
    if api_complexity:
        complexity_dist = Counter()
        for result in task_results:
            for api in result['predicted_apis']:
                complexity = api_complexity.get(api, 'medium')
                complexity_dist[complexity] += 1
        
        print(f"Complexity distribution in predictions:")
        total_preds = sum(complexity_dist.values())
        for complexity, count in sorted(complexity_dist.items()):
            percentage = count / total_preds * 100 if total_preds > 0 else 0
            print(f"  {complexity}: {percentage:.1f}% ({count})")

    return {
        'Micro': {'precision': overall_precision, 'recall': overall_recall, 'f1': 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0},
        'Macro': {'precision': macro_precision, 'recall': macro_recall, 'f1': 2 * macro_precision * macro_recall / (macro_precision + macro_recall) if (macro_precision + macro_recall) > 0 else 0},
        'task_results': task_results,
        'category_stats': dict(category_stats),
        'complexity_stats': dict(complexity_stats),
        'critical_apis_performance': dict(critical_apis_performance),
        'map_score': sum(ranking_scores) / len(ranking_scores) if ranking_scores else 0,
        'coverage_at_k': {k: sum(scores) / len(scores) if scores else 0 for k, scores in coverage_at_k.items()},
        'code_usage_analysis': {
            'total_predicted_used': total_predicted_used,
            'total_predicted_unused': total_predicted_unused,
            'usage_rate': total_predicted_used / total_predicted if total_predicted > 0 else 0,
            'precision_with_usage': tp_and_used / total_predicted_and_used if total_predicted_and_used > 0 else 0,
            'avg_task_usage_rate': sum(task_usage_rates) / len(task_usage_rates) if task_usage_rates else 0,
            'most_unused_apis': dict(predicted_but_unused.most_common(20))
        },
        'llm_knowledge_analysis': {
            'total_llm_knowledge_apis': total_llm_knowledge,
            'tasks_with_llm_knowledge': total_tasks_with_knowledge,
            'llm_knowledge_rate': total_tasks_with_knowledge / len(task_results) if len(task_results) > 0 else 0,
            'avg_llm_knowledge_per_task': sum(r['llm_knowledge_count'] for r in task_results) / len(task_results) if task_results else 0,
            'most_frequent_llm_knowledge_apis': dict(llm_knowledge_apis.most_common(20))
        }
    }

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate API recommendation with comprehensive metrics')
    parser.add_argument('--folder', type=str, default="data/eval_data", help="Path to the folder containing evaluation data")
    parser.add_argument('--gt_file', type=str, help="Path to ground truth API JSON file")
    parser.add_argument('--instruction_file_path', type=str, default="data/rag_data/instructions/scl_brief_keywords.jsonl",
                        help="Path to the instruction file for API recommendation evaluation")
    parser.add_argument('--api_category_file', type=str, help="Path to API category mapping JSON file")
    parser.add_argument('--api_complexity_file', type=str, help="Path to API complexity mapping JSON file")
    
    args = parser.parse_args()

    if not args.gt_file:
        raise ValueError("--gt_file is required for eval_api mode")
    
    print(f"Start evaluating API recommendations in folder: {args.folder}")
    
    instruction_names = []
    if args.instruction_file_path:
        print(f"Using instruction file: {args.instruction_file_path}")
        with open(args.instruction_file_path, 'r', encoding='utf-8') as f:
            instruction_names = [json.loads(line)['instruction_name'] for line in f]
    
    # remove ge、le、lt、gt , since they are not real APIs
    instruction_names = [name for name in instruction_names if not any(op in name.lower() for op in ['ge', 'le', 'lt', 'gt'])]

    res = eval_api_recommendation_advanced(
        args.folder, 
        args.gt_file, 
        instruction_names,
        args.api_category_file,
        args.api_complexity_file
    )

    with open(os.path.join(args.folder, "api_eval_results.json"), 'w', encoding='utf-8') as f:
        json.dump(res, f, indent=4, ensure_ascii=False)

    print("Done.")
