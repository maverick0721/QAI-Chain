import json
import os


def generate_proof():

    # Example inputs
    input_data = {
        "x1": 3,
        "x2": 5,
        "w1": 2,
        "w2": 4
    }

    with open("input.json", "w") as f:
        json.dump(input_data, f)

    # Run Circom (assumes compiled circuit exists)
    os.system("node generate_witness.js ai_inference.wasm input.json witness.wtns")

    # Generate proof
    os.system("snarkjs groth16 prove ai_inference.zkey witness.wtns proof.json public.json")

    print("Proof generated")