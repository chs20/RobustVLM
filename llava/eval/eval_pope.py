import os
import json
import argparse

avg_f1_score = 0

def eval_pope(answers, label_file):
    label_list = [json.loads(q)['label'] for q in open(label_file, 'r')]

    for answer in answers:
        text = answer['text']

        # Only keep the first sentence
        if text.find('.') != -1:
            text = text.split('.')[0]

        text = text.replace(',', '')
        words = text.split(' ')
        if 'No' in words or 'not' in words or 'no' in words:
            answer['text'] = 'no'
        else:
            answer['text'] = 'yes'

    for i in range(len(label_list)):
        if label_list[i] == 'no':
            label_list[i] = 0
        else:
            label_list[i] = 1

    pred_list = []
    for answer in answers:
        if answer['text'] == 'no':
            pred_list.append(0)
        else:
            pred_list.append(1)

    pos = 1
    neg = 0
    yes_ratio = pred_list.count(1) / len(pred_list)

    TP, TN, FP, FN = 0, 0, 0, 0
    for pred, label in zip(pred_list, label_list):
        if pred == pos and label == pos:
            TP += 1
        elif pred == pos and label == neg:
            FP += 1
        elif pred == neg and label == neg:
            TN += 1
        elif pred == neg and label == pos:
            FN += 1

    str_to_log = ''        

    str_to_log += 'TP\tFP\tTN\tFN\t\n'
    str_to_log += '{}\t{}\t{}\t{}\n'.format(TP, FP, TN, FN)

    precision = float(TP) / float(TP + FP)
    recall = float(TP) / float(TP + FN)
    f1 = 2*precision*recall / (precision + recall)
    global avg_f1_score
    avg_f1_score += f1
    acc = (TP + TN) / (TP + TN + FP + FN)
    str_to_log += 'Accuracy: {}\n'.format(acc)
    str_to_log += 'Precision: {}\n'.format(precision)
    str_to_log += 'Recall: {}\n'.format(recall)
    str_to_log += 'F1 score: {}\n'.format(f1)
    str_to_log += 'Yes ratio: {}\n'.format(yes_ratio)
    str_to_log += '%.3f, %.3f, %.3f, %.3f, %.3f\n' % (f1, acc, precision, recall, yes_ratio)
    print(str_to_log)
    return str_to_log

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation-dir", type=str)
    parser.add_argument("--question-file", type=str)
    parser.add_argument("--result-file", type=str)
    parser.add_argument("--model-name", type=str, default='')
    args = parser.parse_args()

    questions = [json.loads(line) for line in open(args.question_file)]
    questions = {question['question_id']: question for question in questions}
    answers = [json.loads(q) for q in open(args.result_file)]
    outputs = ''


    for file in os.listdir(args.annotation_dir):
        assert file.startswith('coco_pope_')
        assert file.endswith('.json')
        category = file[10:-5]
        cur_answers = [x for x in answers if questions[x['question_id']]['category'] == category]
        outputs += 'Category: {}, # samples: {}\n'.format(category, len(cur_answers))
        print('Category: {}, # samples: {}'.format(category, len(cur_answers)))
        outputs += eval_pope(cur_answers, os.path.join(args.annotation_dir, file))
        print("====================================")
    print(f"Average F1-score: {avg_f1_score/3:.4f}")
    with open(f"/data/naman_deep_singh/project_multimodal/pope_evals/{args.model_name}.txt", 'w') as f:
        f.write(outputs)
        f.writelines(f"Average F1-score: {avg_f1_score/3:.4f}\n")