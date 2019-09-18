import math
import argparse
import os.path


# TODO- Read Sigmoid from model file
def sigmoid_transformation\
                (score):
    sigmoid = 1.0
    return 1.0 / (1.0 + math.exp(-1.0 * sigmoid * score))


def get_score(data, model):
    score_list = []

    num_trees = len(model['split_feature'])
    for i in range(0, num_trees):
        node = 0
        while node >= 0:
            if data[model['split_feature'][i][node]] <= model['threshold'][i][node]:
                node = model['left_child'][i][node]
            else:
                node = model['right_child'][i][node]
            if node < 0:
                score = model['leaf_value'][i][abs(node) - 1]
                score_list.append(score)
    return sigmoid_transformation(sum(score_list))


def get_model(model_file):
    model = []
    tree = []
    with open(model_file)as f:
        for line in f:
            if not line.startswith('Tree='):
                tree.append(line.strip().strip('\n'))
            else:
                if 'tree' in tree:
                    tree = []
                    continue
                model.append(tree)
                tree = []
            if line.strip() == "end of trees":
                break

    model.append(tree)
    model_dict = get_model_details(model)
    return model_dict


def get_model_details(model):
    split_feature = []
    threshold = []
    left_child = []
    right_child = []
    leaf_value = []
    model_dict = {}
    for tree in model:
        for item in tree:
            # print(item)
            if 'split_feature' in item:
                sf = [int(x) for x in item.split('=')[1].split(' ')]
                split_feature.append(sf)
            if 'threshold' in item:
                t = [float(x) for x in item.split('=')[1].split(' ')]
                threshold.append(t)
            if 'left_child' in item:
                lc = [int(x) for x in item.split('=')[1].split(' ')]
                left_child.append(lc)
            if 'right_child' in item:
                rc = [int(x) for x in item.split('=')[1].split(' ')]
                right_child.append(rc)
            if 'leaf_value' in item:
                lv = [float(x) for x in item.split('=')[1].split(' ')]
                leaf_value.append(lv)
    model_dict['split_feature'] = split_feature
    model_dict['threshold'] = threshold
    model_dict['left_child'] = left_child
    model_dict['right_child'] = right_child
    model_dict['leaf_value'] = leaf_value

    return model_dict


def score_data(data_file, model, delimit):
    # Read the data
    score_list_tree = []
    with open(data_file) as f:
        for line in f:
            data = [float(x) for x in line.strip('\n').split(delimit)]
            score = get_score(data, model)
            score_list_tree.append(score)

    return score_list_tree


def write_result(out_file, score_list):
    # Write results to a output file
    with open(out_file, 'a+') as o_file:
        for item in score_list:
            o_file.write("%s\n" % item)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", help="Data file to be predicted", required=True)
    parser.add_argument("-m", help="Model file to be used for prediction", required=True)
    parser.add_argument("-o", help="Output file for writing results", required=True)
    args = parser.parse_args()

    data_file = args.d
    model_file = args.m
    out_file = args.o

    if not os.path.isfile(data_file):
        print("Data file " + str(data_file) + " does not exist. Exiting.....!!")
        return

    if not os.path.isfile(model_file):
        print("Model file " + str(model_file) + " does not exist. Exiting.....!!")
        return

    model = get_model(model_file)

    score_list_tree = score_data(data_file, model, "\t")

    write_result(out_file, score_list_tree)


if __name__ == "__main__":
    main()
