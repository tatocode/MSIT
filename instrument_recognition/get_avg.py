import os

if __name__ == "__main__":
    models = ["vgg19bn", "densenet201", "resnet50", "resnet101", "vit", "swin", "banet_r50", "banet_r101"]

    for mls in models:
        path_1 = os.path.join("result", f"{mls}_train")
        path_2 = os.path.join(path_1, sorted(os.listdir(path_1))[-1])
        for fold in range(4):
            print(f"model: {mls}, fold: {fold}")
            path = os.path.join(path_2, str(fold), "log.txt")
            with open(path, "r") as rf:
                cts = rf.readlines()
            for c in cts[1::2]:
                print(c.strip()[-4:])
            break
        break
