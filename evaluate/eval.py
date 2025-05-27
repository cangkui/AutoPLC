import os
import time
import json
import requests
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Set
from tqdm import tqdm


from logging import getLogger
logger = getLogger("evaluate.eval")

@dataclass
class ErrorMessage:
    error_desc: str
    error_type: str
    code_window: Optional[str] = None

    def to_dict(self):
        return {
            "error_desc": self.error_desc,
            "error_type": self.error_type,
            "code_window": self.code_window
        }

    @classmethod
    def from_dict(cls, data: dict):
        return cls(
            error_desc=data['error_desc'],
            error_type=data.get('error_type', '代码段错误'),
            code_window=data.get('code_window')
        )


@dataclass
class ResponseData:
    success: bool
    result: Optional[str] = None
    errors: List[ErrorMessage] = None

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "result": self.result,
            "errors": [e.to_dict() for e in self.errors] if self.errors else []
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    @classmethod
    def from_dict(cls, data: dict):
        errors = data.get("errors", [])
        return cls(
            success=data["success"],
            result=data.get("result"),
            errors=[ErrorMessage.from_dict(e) for e in errors]
        )

    @classmethod
    def default_false(cls):
        return cls(
            success=False,
            result=None,
            errors=[ErrorMessage(
                error_desc="编译工具调用失败",
                error_type="系统错误"
            )]
        )


class CodesysCompiler:
    def extract_code_window(self, source_code: str, error_info: Dict, window_size: int = 3) -> str:
        lines = source_code.splitlines()
        path = error_info.get("Path", 0)
        is_def = error_info.get("IsDef", False)

        base_line_idx = 0
        if not is_def:
            begin_idx = next((i for i, line in enumerate(lines) if "BEGIN" in line.upper()), None)
            if begin_idx is not None:
                base_line_idx = begin_idx
            else:
                end_var_indices = [i for i, line in enumerate(lines) if "END_VAR" in line.upper()]
                base_line_idx = end_var_indices[-1] + 1 if end_var_indices else 0

        error_line_idx = base_line_idx + path
        start_idx = max(0, error_line_idx - window_size)
        end_idx = min(len(lines), error_line_idx + window_size + 1)

        return "\n".join(f"{i + 1:>4}: {lines[i]}" for i in range(start_idx, end_idx))

    def syntax_check(self, block_name: str, st_code: str) -> ResponseData:
        API_KEY = "admin"  # Default API key, change in production
        # Configure requests session
        session = requests.Session()
        session.headers.update({
            'Authorization': 'ApiKey ' + API_KEY,
            'Content-Type': 'application/json'
        })
        URL = "http://192.168.103.117:9000/api/v1/pou/workflow"
        json_data = {"BlockName": block_name, "Code": st_code}
        timeout = 30  # Set a reasonable timeout for the request
        try:
            resp = session.post(URL, json=json_data, timeout=timeout)  # Reasonable timeout
            # resp = requests.post(
            #     url="http://192.168.103.117:9000/api/v1/pou/workflow",
            #     json={"BlockName": block_name, "Code": st_code}
            # )
            print(resp.json())
        
            if resp.status_code != 200:
                return ResponseData.default_false()

            raw_data = resp.json()
            raw_errors = raw_data.get("Errors", [])
            simplified_errors = []

            for err in raw_errors:
                code_window = self.extract_code_window(st_code, err, window_size=3)
                simplified_errors.append(ErrorMessage(
                    error_desc=err["ErrorDesc"],
                    error_type="定义区错误" if err.get("IsDef", False) else "代码段错误",
                    code_window=code_window
                ))

            return ResponseData(
                success=raw_data.get("Success", True),
                result=raw_data.get("Result", ""),
                errors=simplified_errors
            )
        except Exception as e:
            print(f"[Error] Codesys Compiler API failed: {e}")
            return ResponseData.default_false()


    def batch_test_exp_folder(self, exp_folder: str, save_dir: str):
        os.makedirs(save_dir, exist_ok=True)
        for case_name in os.listdir(exp_folder):
            case_path = os.path.join(exp_folder, case_name)
            if not os.path.isdir(case_path):
                continue

            st_files = [f for f in os.listdir(case_path) if f.endswith('.st')]
            if not st_files:
                print(f"[WARN] No ST file in {case_path}")
                continue

            st_file = st_files[0]
            st_path = os.path.join(case_path, st_file)

            with open(st_path, 'r', encoding='utf-8') as f:
                scl_code = f.read()

            print(f"Testing case: {case_name}")
            response = self.syntax_check(case_name, scl_code)

            # 保存结果
            exp_folder_name = os.path.basename(os.path.abspath(exp_folder))
            save_name = f"{exp_folder_name}_{case_name}.json"
            save_path = os.path.join(save_dir, save_name)

            with open(save_path, 'w', encoding='utf-8') as out_f:
                out_f.write(response.to_json())

class TIAPortalCompiler:
    def extract_code_window(self, source_code: str, error_info: Dict, window_size: int = 3) -> str:
        lines = source_code.splitlines()
        path = error_info.get("Path", 0)
        is_def = error_info.get("IsDef", False)

        base_line_idx = 0
        if not is_def:
            begin_idx = next((i for i, line in enumerate(lines) if "BEGIN" in line.upper()), None)
            if begin_idx is not None:
                base_line_idx = begin_idx
            else:
                end_var_indices = [i for i, line in enumerate(lines) if "END_VAR" in line.upper()]
                base_line_idx = end_var_indices[-1] + 1 if end_var_indices else 0

        error_line_idx = base_line_idx + path
        start_idx = max(0, error_line_idx - window_size)
        end_idx = min(len(lines), error_line_idx + window_size + 1)

        return "\n".join(f"{i + 1:>4}: {lines[i]}" for i in range(start_idx, end_idx))

    def syntax_check(self, block_name: str, scl_code: str) -> ResponseData:
        try:
            resp = requests.post(
                url="http://192.168.103.152:9000/api/tiaapi/process",
                json={"BlockName": block_name, "Code": scl_code}
            )
            if resp.status_code != 200:
                return ResponseData.default_false()
            raw_data = resp.json()
            raw_errors = raw_data.get("Errors", [])
            simplified_errors = []

            for err in raw_errors:
                code_window = self.extract_code_window(scl_code, err, window_size=3)
                simplified_errors.append(ErrorMessage(
                    error_desc=err["ErrorDesc"],
                    error_type="定义区错误" if err.get("IsDef", False) else "代码段错误",
                    code_window=code_window
                ))

            return ResponseData(
                success=raw_data.get("Success", True),
                result=raw_data.get("Result", ""),
                errors=simplified_errors
            )
        except Exception as e:
            print(f"[Error] TIA Compiler API failed: {e}")
            return ResponseData.default_false()

    def batch_test_exp_folder(self, exp_folder: str, save_dir: str):
        os.makedirs(save_dir, exist_ok=True)
        for case_name in os.listdir(exp_folder):
            case_path = os.path.join(exp_folder, case_name)
            if not os.path.isdir(case_path):
                continue

            scl_files = [f for f in os.listdir(case_path) if f.endswith('.scl')]
            if not scl_files:
                print(f"[WARN] No SCL file in {case_path}")
                continue

            scl_file = scl_files[0]
            scl_path = os.path.join(case_path, scl_file)

            with open(scl_path, 'r', encoding='utf-8') as f:
                scl_code = f.read()

            print(f"Testing case: {case_name}")
            response = self.syntax_check(case_name, scl_code)

            # 保存结果
            exp_folder_name = os.path.basename(os.path.abspath(exp_folder))
            save_name = f"{exp_folder_name}_{case_name}.json"
            save_path = os.path.join(save_dir, save_name)

            with open(save_path, 'w', encoding='utf-8') as out_f:
                out_f.write(response.to_json())



def eval_result(folder_path, compiler_type='codesys'):
    from tqdm import tqdm

    if compiler_type == 'codesys':
        compiler = CodesysCompiler()
        file_prefix = "st"
    elif compiler_type == 'tiaportal':
        compiler = TIAPortalCompiler()
        file_prefix = "scl"
    else:
        raise ValueError("Unsupported compiler type. Use 'codesys' or 'tiaportal'.")
    
    py_dir = os.path.dirname(os.path.abspath(__file__))

    # 创建 compile_results/{exp_name}/
    exp_name = os.path.basename(os.path.normpath(folder_path))
    save_root = os.path.join(py_dir, "compile_results")
    save_dir = os.path.join(save_root, exp_name)
    os.makedirs(save_dir, exist_ok=True)

    # 预处理出所有文件
    case_list = []
    for case_name in os.listdir(folder_path):
        case_path = os.path.join(folder_path, case_name)
        if os.path.isdir(case_path):
            scl_files = [f for f in os.listdir(case_path) if f.endswith(f".{file_prefix}")]
            if scl_files:
                case_list.append((case_name, os.path.join(case_path, scl_files[0])))

    total_cases = len(case_list)
    success_cases = 0
    failed_cases = 0
    total_errors = 0
    failed_files = []

    for case_name, scl_path in tqdm(case_list, desc=f"Compiling ({exp_name})"):
        with open(scl_path, 'r', encoding='utf-8') as f:
            st_code = f.read()

        response = compiler.syntax_check(case_name, st_code)

        if response.success:
            success_cases += 1
        else:
            failed_cases += 1
            failed_files.append(case_name)
            total_errors += len(response.errors or [])

        # 保存每个测试样例的 JSON
        save_path = os.path.join(save_dir, f"{case_name}.json")
        with open(save_path, 'w', encoding='utf-8') as out_f:
            out_f.write(response.to_json())

    # 汇总信息
    pass_rate = (success_cases / total_cases) * 100 if total_cases else 0
    avg_errors = (total_errors / failed_cases) if failed_cases else 0

    summary = {
        "total_cases": total_cases,
        "success_cases": success_cases,
        "failed_cases": failed_cases,
        "pass_rate_percent": round(pass_rate, 2),
        "total_errors": total_errors,
        "avg_errors_per_failed_case": round(avg_errors, 2),
        "failed_files": failed_files
    }

    print(f"Compiler {compiler_type.capitalize()} Results for {exp_name}:")
    print("\n=== Compilation Summary ===")
    for k, v in summary.items():
        print(f"{k}: {v}")

    # 保存统计信息
    summary_path = os.path.join(save_dir, f"{exp_name}_summary.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def eval_api_recommendation(task_folder: str, ground_truth_path: str):
    with open(ground_truth_path, 'r', encoding='utf-8') as f:
        gt_data = json.load(f)
    gt_data = {k.lower(): v for k, v in gt_data.items()}  # 统一 key 为小写

    total_tp, total_fp, total_fn = 0, 0, 0
    task_names = [name for name in os.listdir(task_folder) if os.path.isdir(os.path.join(task_folder, name))]
    max_gt_len = 0
    max_pred_len = 0
    for task in tqdm(task_names, desc="Evaluating API Recommendations"):
        inter_path = os.path.join(task_folder, task, "intermediate_results.json")
        if not os.path.exists(inter_path):
            continue

        with open(inter_path, 'r', encoding='utf-8') as f:
            inter_data = json.load(f)
        predicted: Set[str] = set(api.lower() for api in inter_data.get("apis_for_this_task", []))
        actual: Set[str] = set(api.lower() for api in gt_data.get(task.lower(), []) if "_TO_" not in api)

        if len(actual) > max_gt_len:
            max_gt_len = len(actual)
        if len(predicted) > max_pred_len:
            max_pred_len = len(predicted)

        tp = len(predicted & actual)
        fp = len(predicted - actual)
        fn = len(actual - predicted)

        # 打印没有找到的 API
        # if fn > 0:
        #     print(f"\nTask: {task}")
        #     print(f"Missing APIs: {actual - predicted}")

        total_tp += tp
        total_fp += fp
        total_fn += fn

    precision = total_tp / (total_tp + total_fp) if total_tp + total_fp else 0
    recall = total_tp / (total_tp + total_fn) if total_tp + total_fn else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0

    print("\n=== API Recommendation Evaluation Summary ===")
    print(f"Total TP: {total_tp}, FP: {total_fp}, FN: {total_fn}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"Max GT Length: {max_gt_len}")
    print(f"Max Predicted Length: {max_pred_len}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate SCL compilation or API recommendation')
    parser.add_argument('--mode', type=str, choices=['compile', 'eval_api'], default='compile')
    parser.add_argument('--folder', type=str, default="data/eval_data")
    parser.add_argument('--gt_file', type=str, help="Path to ground truth API JSON file")
    parser.add_argument('--compiler', type=str, choices=['codesys', 'tiaportal'], default='codesys',
                        help="Compiler type to use for evaluation")
    args = parser.parse_args()

    if args.mode == 'compile':
        print(f"Start compiling folder: {args.folder}")
        eval_result(folder_path=args.folder, compiler_type=args.compiler)
    elif args.mode == 'eval_api':
        if not args.gt_file:
            raise ValueError("--gt_file is required for eval_api mode")
        eval_api_recommendation(args.folder, args.gt_file)

    print("Done.")
