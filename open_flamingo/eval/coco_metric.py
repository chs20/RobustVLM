from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.eval import COCOEvalCap
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocotools.coco import COCO


def compute_cider(
    result_path,
    annotations_path,
):
    # create coco object and coco_result object
    coco = COCO(annotations_path)
    coco_result = coco.loadRes(result_path)

    # create coco_eval object by taking coco and coco_result
    coco_eval = COCOEvalCap(coco, coco_result)
    coco_eval.params["image_id"] = coco_result.getImgIds()
    coco_eval.evaluate()

    return coco_eval.eval

def compute_cider_all_scores(
    result_path,
    annotations_path,
    return_img_ids=False,
):
    # create coco object and coco_result object
    coco = COCO(annotations_path)
    coco_result = coco.loadRes(result_path)

    cider_scorer = Cider()
    imgIds = coco_result.getImgIds()
    gts = {}
    res = {}
    for imgId in imgIds:
        gts[imgId] = coco.imgToAnns[imgId]
        res[imgId] = coco_result.imgToAnns[imgId]
    tokenizer = PTBTokenizer()
    gts = tokenizer.tokenize(gts)
    res = tokenizer.tokenize(res)
    score, scores = cider_scorer.compute_score(gts, res)
    scores *= 100
    if return_img_ids:
        return scores, imgIds
    else:
        return scores

def postprocess_captioning_generation(predictions):
    return predictions.split("Output", 1)[0]

if __name__ == '__main__':
    result_path = "/mnt/cschlarmann37/project_multimodal/llava-evals/captions-json/cocoresults_38eb6f53-71e4-469e-a864-cb64b1fdbbf4.json"
    annotations_path = "/mnt/datasets/coco/annotations/captions_val2014.json"
    print(f"\nresult_path: {result_path}\n")
    metrics = compute_cider(result_path, annotations_path)
    print(metrics)
    print(f"CIDER: {metrics['CIDEr']*100}")