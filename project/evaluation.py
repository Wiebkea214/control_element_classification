from project.main import *

#############################################################################

if __name__ == "__main__":

    base_dir = Path(__file__).resolve().parent
    evaluation = 0

    if evaluation:
        # Automatic evaluation execution
        top_xs = [3, 5, 7, 10, 10000]
        feats = [0, 9]
        classes = [4, 8, 12, 16, 20, 24, 28]
        kernels = ["linear", "poly"]

        # Available configs: load_fts, edit_db, similarity, train_svm, train_svm_only, evaluate
        config_x = "train_svm, evaluate_model"

        # Current values
        cab_x = ""
        class_x = 28
        top_x = 5
        feat_x = 9
        cnt = 250
        kernel_x = "linear"
        c_x = 10
        test_step = ""

        path_train_x = f"F:\\OneDrive\\Masterarbeit\\FTS Daten\\Training\\TRAXX_AC3_Training_allCabs_{class_x}class_cnt{cnt}.xlsx"
        dir_name_x = f"evaluation_allCabs_top{top_x}_{class_x}class_{feat_x}feat_{kernel_x}Kernel_cnt{cnt}"
        print(f"----- Start with param feat_x={feat_x}, top_xs={top_x}, class={class_x}, c={c_x}, cnt={cnt} -----")

        main(cab_x, top_x, feat_x, kernel_x, c_x, path_train_x, dir_name_x, config_x, test_step)

    else:
        # Normal operation parameters
        project = "AC3"
        class_x = 28
        cnt = 250

        # Interface parameters
        cab_x = "cab1"
        config_x = "predict"
        test_step_x = "Switch the battery on using the pushbutton on rearwall"

        # Paths
        dir_train = base_dir / "Train data"
        path_train = dir_train / f"TRAXX_{project}_Training_allCabs_{class_x}class_cnt{cnt}.xlsx"

        prediction = main(cab_x, k=5, feat=9, kernel="linear", c=10, path_train=path_train, dir_name="",
                          config=config_x, test_step=test_step_x)