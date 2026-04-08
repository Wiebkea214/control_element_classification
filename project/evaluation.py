from project.main import *

#############################################################################

if __name__ == "__main__":

    base_dir = Path(__file__).resolve().parent

    # Automatic evaluation execution
    top_xs = [3, 5, 7, 10, 10000]
    feats = [0, 9]
    classes = [4, 8, 12, 16, 20, 24, 28]
    kernels = ["linear", "poly"]

    # Available configs: load_fts, edit_db, similarity, train_svm, train_svm_only, evaluate
    config_x = "train_svm, evaluate_model"

    # Current values
    cab_x = ""
    class_x = 8
    top_x = 3
    feat_x = 9
    cnt = 150
    kernel_x = "linear"
    c_x = 1
    test_step = ""

    for feat_x in feats:
        path_train_x = base_dir / "Train data" / f"TRAXX_AC3_Training_allCabs_{class_x}class_cnt{cnt}.xlsx"
        dir_name_x = f"evaluation_allCabs_top{top_x}_{class_x}class_{feat_x}feat_{kernel_x}Kernel_cnt{cnt}"
        print(f"----- Start with param feat_x={feat_x}, top_xs={top_x}, class={class_x}, c={c_x}, cnt={cnt} -----")

        main(cab_x, top_x, feat_x, kernel_x, c_x, path_train_x, dir_name_x, config_x, test_step)
