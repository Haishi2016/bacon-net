"""Generate LogicCBM-format pkls (312-dim per-image attributes) from our CUB data,
with correct local Windows paths. Sets val.pkl = test.pkl so the reference code's
validation loop reports accuracy on OUR official test set (apples-to-apples)."""
import os, pickle

CUB = r"C:\School\datasets\cub\CUB_200_2011"
OUT = r"C:\School\datasets\cub\CUB_logicref_processed"
os.makedirs(OUT, exist_ok=True)

# uncertainty calibration (Koh data_processing.py)
UMAP = {1: {1: 0, 2: 0.5, 3: 0.75, 4: 1}, 0: {1: 0, 2: 0.5, 3: 0.25, 4: 0}}

# per-image 312 attribute_label + certainty from the raw annotation file
attr, cert = {}, {}
with open(os.path.join(CUB, "attributes", "image_attribute_labels.txt")) as f:
    for line in f:
        p = line.split()
        if len(p) < 4:
            continue
        iid, aid, lab, c = int(p[0]), int(p[1]), int(p[2]), int(p[3])
        attr.setdefault(iid, [0] * 312)[aid - 1] = lab
        cert.setdefault(iid, [0] * 312)[aid - 1] = c


def _local(img_path):
    s = img_path.replace("\\", "/")
    i = s.find("images/")
    return (CUB.replace("\\", "/") + "/" + s[i:])


def convert(split_pkl):
    with open(os.path.join(CUB, split_pkl), "rb") as f:
        entries = pickle.load(f)
    out = []
    for e in entries:
        iid = e["id"]
        a = attr[iid]; c = cert[iid]
        ua = [UMAP[a[j]][c[j]] for j in range(312)]
        out.append({"id": iid, "img_path": _local(e["img_path"]),
                    "class_label": e["class_label"], "attribute_label": a,
                    "attribute_certainty": c, "uncertain_attribute_label": ua})
    return out


train = convert("train.pkl")
test = convert("test.pkl")
for name, data in [("train", train), ("val", test), ("test", test)]:
    with open(os.path.join(OUT, f"{name}.pkl"), "wb") as f:
        pickle.dump(data, f)
    print(f"{name}.pkl: {len(data)} entries, attr_dim {len(data[0]['attribute_label'])}", flush=True)
print("sample img_path:", train[0]["img_path"])
print("exists:", os.path.exists(train[0]["img_path"]))
