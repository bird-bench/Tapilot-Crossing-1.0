import sys
from io import StringIO
import contextlib
import json
import os
import contextlib
import openai
import numpy as np

def is_penultimate_directory(path):  
    """  
    This function checks if the given path is a penultimate directory (i.e., its subdirectories do not contain any other directories).  
  
    :param path: Path to the directory to check  
    :return: True if the directory is penultimate, False otherwise  
    """  
    # List all subdirectories in the given path  
    subdirectories = [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]  
  
    # If there are no subdirectories, the given path is not a penultimate directory  
    if not subdirectories:  
        return False  
  
    # Check each subdirectory to see if it contains any other directories  
    for subdir in subdirectories:  
        subdir_path = os.path.join(path, subdir)  
        subdir_subdirectories = [name for name in os.listdir(subdir_path) if os.path.isdir(os.path.join(subdir_path, name))]  
  
        # If a subdirectory contains other directories, the given path is not a penultimate directory  
        if subdir_subdirectories:  
            return False  
  
    # If none of the subdirectories contain other directories, the given path is a penultimate directory  
    return True 

def get_embedding(text):
    response = openai.Embedding.create(
      engine="text-embedding-ada-002",
      input=[text],
      max_tokens=0
    )
    
    return response['data'][0]['embedding']

@contextlib.contextmanager  
def capture_output():  
    new_out, new_err = StringIO(), StringIO()  
    old_out, old_err = sys.stdout, sys.stderr  
    try:  
        sys.stdout, sys.stderr = new_out, new_err  
        yield sys.stdout, sys.stderr  
    finally:  
        sys.stdout, sys.stderr = old_out, old_err 

# Path to the JSON file containing the answers
json_path = "tapilot_data/src/my_llm_response/normal/All_normal_baseline_gpt4_COT.json"
seg_fn = "pred_code_segment_gpt_4_trubo_COT.py"
combine_fn = "pred_code_gpt_4_turbo_COT.py"

with open(json_path, "r") as f_json:
    llm_response = json.load(f_json)
results = []
total_questions = 0
for root, llm_resp in llm_response.items(): 
    code_base_path = os.path.join(root, 'src/ref_code_hist.py')
    with open(code_base_path, "r") as f_code:
        code_base = f_code.read()

    with open(os.path.join(root, 'prompt_curr.txt'), "r") as f_p:
        prompt_curr = f_p.read()

    cut_idx = prompt_curr.rfind("[YOU (AI assistant)]")
    if cut_idx != -1:
        prompt_curr = prompt_curr[cut_idx:]

    cut_idx_2 = prompt_curr.rfind("'''")
    if cut_idx_2 != -1:
        code_head = prompt_curr[cut_idx_2+3:]
    else:
        raise AssertionError
    
    if "---BEGIN " in llm_resp:
        cut_idx = llm_resp.find("---BEGIN ")
        code_gen = llm_resp[cut_idx:]
        
        cut_idx_2 = code_gen.find("import ")
        if "from " in code_gen[:cut_idx_2]:
            print(root)
            raise AssertionError
        code_add = code_gen[cut_idx_2:]

        cut_idx_3 = code_add.rfind("```")
        if cut_idx_3 == -1:
            cut_idx_3 = code_add.rfind("---END ")
            if cut_idx_3 == -1:
                cut_idx_3 = code_add.rfind("'''")
        code_add = code_add[:cut_idx_3]

    elif "```python" in llm_resp:
        cut_idx = llm_resp.find("```python")
        code_gen = llm_resp[cut_idx:].replace("```python", "")
        
        cut_idx_2 = code_gen.find("import ")
        if "from " in code_gen[:cut_idx_2]:
            print(root)
            raise AssertionError
        code_add = code_gen[cut_idx_2:]
        cut_idx_3 = code_add.rfind("```")
        if cut_idx_3 == -1:
            cut_idx_3 = code_add.rfind("---END CODE TEMPLATE---")
            if cut_idx_3 == -1:
                cut_idx_3 = code_add.rfind("'''")
        code_add = code_add[:cut_idx_3]

    else:
        if "import " in llm_resp:
            code_gen = llm_resp
            cut_idx_2 = code_gen.find("import ")

            code_add = code_gen[cut_idx_2:]
            cut_idx = code_add.rfind("```")
            if cut_idx != -1:
                code_add = code_add[:cut_idx]
            
            code_add = code_add.replace("\n'''", "").replace("\n---END CODE TEMPLATE---", "").replace("\n```", "")
        else:
            cut_idx = llm_resp.rfind("'''")
            code_add = code_head + llm_resp[:cut_idx]
            code_add = code_add.replace("'''", "").replace("---END CODE TEMPLATE---", "").replace("```", "")
    
    code_add = code_add.replace("---END CODE TEMPLATE---", "").replace("```", "")
    cut_idx = code_add.rfind("pickle.dump(")
    if cut_idx == -1:
        cut_idx = code_add.rfind("plt.")
        if cut_idx == -1:
            cut_idx = code_add.rfind("save_plot(")
            cut_idx_tmp = code_add.rfind("save_plot(filename, dpi=100")
            if cut_idx == cut_idx_tmp:
                cut_idx = len(code_add) - int(np.ceil(len(code_add)/50))

    code_end = code_add[cut_idx:]
    code_end_clr = code_end.replace("'''", "")
    code_add = code_add.replace(code_end, code_end_clr)
    change_flg = False
    for line in code_add.split("\n"):
        if "pickle.load(" in line:
            code_add = code_add.replace(line, "")
            
    if "meta_" not in root:
        if "turn_1" not in root:
            for line in code_add.split("\n"):
                if "def read_csv_file(" in line:
                    continue
                if "read_csv_file(" in line or "pd.read_csv(" in line:
                    code_add = code_add.replace(line, "")
        else:
            if "tapilot_data/atp_tennis.csv" not in code_add:
                code_add = code_add.replace("ATP_tennis.csv", "tapilot_data/atp_tennis.csv")
            if "tapilot_data/credit_customers.csv" not in code_add:
                code_add = code_add.replace("credit_customers.csv", "tapilot_data/credit_customers.csv")
            if "tapilot_data/fastfood.csv" not in code_add:
                code_add = code_add.replace("fastfood.csv", "tapilot_data/fastfood.csv")
            if "tapilot_data/laptops_price.csv" not in code_add:
                code_add = code_add.replace("laptops_price.csv", "tapilot_data/laptops_price.csv")
            if "tapilot_data/melb_data.csv" not in code_add:
                code_add = code_add.replace("melb_data.csv", "tapilot_data/melb_data.csv")
    else:
        if "turn_1_meta_1" not in root:
            for line in code_add.split("\n"):
                if "def read_csv_file(" in line:
                    continue
                if "read_csv_file(" in line or "pd.read_csv(" in line:
                    code_add = code_add.replace(line, "")
        else:
            if "tapilot_data/atp_tennis.csv" not in code_add:
                code_add = code_add.replace("ATP_tennis.csv", "tapilot_data/atp_tennis.csv")
            if "tapilot_data/credit_customers.csv" not in code_add:
                code_add = code_add.replace("credit_customers.csv", "tapilot_data/credit_customers.csv")
            if "tapilot_data/fastfood.csv" not in code_add:
                code_add = code_add.replace("fastfood.csv", "tapilot_data/fastfood.csv")
            if "tapilot_data/laptops_price.csv" not in code_add:
                code_add = code_add.replace("laptops_price.csv", "tapilot_data/laptops_price.csv")
            if "tapilot_data/melb_data.csv" not in code_add:
                code_add = code_add.replace("melb_data.csv", "tapilot_data/melb_data.csv")

    code_add = code_add.replace("</code>", "").replace("<code>", "")

    with open(os.path.join(root, seg_fn), "w") as f_out:
        f_out.write(code_add)

    code_all = code_base + "\n\n" + code_add
    for line in code_all.split("\n"):
        if "plt.show(" in line or "show_plots(" in line:
            code_all = code_all.replace(line, "")

    with open(os.path.join(root, combine_fn), "w") as f_out:
        f_out.write(code_all)
