"""
generate_train_rou.py

A reusable module for generating .rou.xml files with variable traffic flow for SUMO + DRL training pipelines.
Can be directly imported and called in training scripts.
"""

import os
import random
from pathlib import Path

VTYPE_DEFINITIONS = """<vType id="car" vClass="passenger" accel="2.6" decel="4.5" sigma="0.5" length="5.0" maxSpeed="13.9"/>
<vType id="motorcycle" vClass="motorcycle" accel="3.0" decel="5.0" sigma="0.5" length="2.0" maxSpeed="14.0"/>
<vType id="truck" vClass="truck" accel="1.2" decel="3.5" sigma="0.5" length="12.0" maxSpeed="10.0"/>
"""

EXIT_OPTIONS = {
    "222682222#1": {0: "193253427#0", 1: "198876043#0", 2: "28231434",   3: "23241821#0", 4: "198876042#0"},
    "193253433#2": {0: "198876043#0", 1: "28231434",   2: "23241821#0", 3: "198876042#0", 4: "193253427#0"},
    "198876045#1": {0: "28231434",    1: "23241821#0", 2: "198876042#0", 3: "193253427#0", 4: "198876043#2"},
    "-28231434":   {0: "23241821#0",  1: "198876042#0", 2: "193253427#0", 3: "198876043#2", 4: "28231434"},
    "-55029244#0": {0: "198876042#0", 1: "193253427#0", 2: "198876043#2", 3: "28231434",    4: "23241821#0"},
    "34241530#0":  {0: "198876042#0", 1: "193253427#0", 2: "198876043#2", 3: "28231434",    4: "23241821#0"},
}

FLOW_DEFINITIONS_INDEX_BASED = [
    ("car","222682222#1",0,0.01296),("car","222682222#1",1,0.0108),("car","222682222#1",2,0.0108),
    ("car","222682222#1",3,0.00864),("car","222682222#1",4,0.0072),
    ("motorcycle","222682222#1",0,0.03744),("motorcycle","222682222#1",1,0.03312),
    ("motorcycle","222682222#1",2,0.03024),("motorcycle","222682222#1",3,0.02448),
    ("motorcycle","222682222#1",4,0.0216),
    ("car","193253433#2",0,0.00864),("car","193253433#2",1,0.0072),("car","193253433#2",2,0.0072),
    ("car","193253433#2",3,0.00576),("car","193253433#2",4,0.00504),
    ("motorcycle","193253433#2",0,0.02448),("motorcycle","193253433#2",1,0.0216),
    ("motorcycle","193253433#2",2,0.02016),("motorcycle","193253433#2",3,0.01728),
    ("motorcycle","193253433#2",4,0.0144),
    ("car","198876045#1",0,0.0072),("car","198876045#1",1,0.00648),("car","198876045#1",2,0.00576),
    ("car","198876045#1",3,0.00504),("car","198876045#1",4,0.00432),
    ("motorcycle","198876045#1",0,0.0216),("motorcycle","198876045#1",1,0.01872),
    ("motorcycle","198876045#1",2,0.01728),("motorcycle","198876045#1",3,0.01584),
    ("motorcycle","198876045#1",4,0.0144),
    ("car","-28231434",0,0.00504),("car","-28231434",1,0.00432),("car","-28231434",2,0.0036),
    ("car","-28231434",3,0.0036),("car","-28231434",4,0.00288),
    ("motorcycle","-28231434",0,0.0144),("motorcycle","-28231434",1,0.01296),
    ("motorcycle","-28231434",2,0.01152),("motorcycle","-28231434",3,0.01008),
    ("motorcycle","-28231434",4,0.00864),
    ("car","-55029244#0",0,0.00576),("car","-55029244#0",1,0.00504),("car","-55029244#0",2,0.00432),
    ("car","-55029244#0",3,0.0036),("car","-55029244#0",4,0.00288),
    ("motorcycle","-55029244#0",0,0.01728),("motorcycle","-55029244#0",1,0.0144),
    ("motorcycle","-55029244#0",2,0.01296),("motorcycle","-55029244#0",3,0.01152),
    ("motorcycle","-55029244#0",4,0.01008),
    ("car","34241530#0",0,0.00576),("car","34241530#0",1,0.00504),("car","34241530#0",2,0.00432),
    ("car","34241530#0",3,0.0036),("car","34241530#0",4,0.00288),
    ("motorcycle","34241530#0",0,0.01728),("motorcycle","34241530#0",1,0.0144),
    ("motorcycle","34241530#0",2,0.01296),("motorcycle","34241530#0",3,0.01152),
    ("motorcycle","34241530#0",4,0.01008),
    ("truck","34241530#0",0,0.00252),
]

def generate_training_route_with_variation(
    path=None,
    seed=42,
    end_time=3600,
    demand_scaling=1.0,
    noise_level=0.1
):
    if path is None:
        path = str(Path(__file__).resolve().parent / "cfg" / f"train_flow_{seed:08x}.rou.xml")
    os.makedirs(os.path.dirname(path), exist_ok=True)

    rnd = random.Random(seed)

    with open(path, "w") as f:
        f.write("<routes>\n")
        f.write(VTYPE_DEFINITIONS + "\n")

        for vtype, frm, idx, prob_sec in FLOW_DEFINITIONS_INDEX_BASED:
            to = EXIT_OPTIONS.get(frm, {}).get(idx)
            if not to:
                continue
            noise_factor = rnd.uniform(1 - noise_level, 1 + noise_level)
            vph = prob_sec * 3600.0 * demand_scaling * noise_factor
            if vph < 1.0:
                continue
            flow_id = f"{vtype}_{frm}_{to}_{seed:08x}"
            f.write(
                f'  <flow id="{flow_id}" type="{vtype}" begin="0" end="{end_time}" '
                f'vehsPerHour="{vph:.2f}" departLane="best" departPos="random" '
                f'departSpeed="0" from="{frm}" to="{to}"/>\n'
            )
        f.write("</routes>\n")

    print(f"[✔] Generated training flow with variation: {path}")
    return path, seed

if __name__ == "__main__":
    import numpy as np
    SEED = np.random.randint(0, 1_000_000)
    _ = generate_training_route_with_variation(seed=SEED, demand_scaling=1.0, noise_level=0.1)