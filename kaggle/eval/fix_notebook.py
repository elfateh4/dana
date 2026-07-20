import json
with open("06_solver_ml.ipynb", "r") as f:
    nb = json.load(f)
for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        source = cell["source"]
        if isinstance(source, list) and any("=== POMO ===" in line for line in source):
            new_source = []
            for line in source:
                if "m_vrptw = load_model(ckpt_pomo_vrptw)" in line:
                    if not line.startswith("    "):
                        line = "    " + line
                if "res = run_inference(td, model, coords, n_starts=8)" in line:
                    line = line.replace("model", "m_cvrp if inst.get('problem_type')=='cvrp' else m_vrptw")
                new_source.append(line)
            cell["source"] = new_source
with open("06_solver_ml.ipynb", "w") as f:
    json.dump(nb, f, indent=1)
print("Fixed notebook.")
