import json


def get_box_from_ocr_box(ocr_box):
    """
    Returns the bounding box in the format (xmin, ymin, xmax, ymax)
    """
    xs = [vertex.x for vertex in ocr_box.vertices]
    ys = [vertex.y for vertex in ocr_box.vertices]
    return (min(xs), min(ys), max(xs), max(ys))


class GroundedLegendValue:
    def __init__(self, value, ocr_boxes):
        self.value = value
        self.ocr_boxes = ocr_boxes
        self.translated_value = None
        self.simplified_value = None

    def add_box(self, box):
        self.ocr_boxes.append(box)
        
    def set_translated_value(self, translated_value):
        self.translated_value = translated_value

    def set_simplified_value(self, simplified_value):
        self.simplified_value = simplified_value
        
    


class GroundedLegend:
    def __init__(self):
        self.legend = {}

    def update_legend_value(self, key, value, ocr_box):
        if key in self.legend:
            self.legend[key].add_box(ocr_box)
        else:
            legend_item = GroundedLegendValue(value, [ocr_box])
            self.legend[key] = legend_item

    def to_json(self, output_path):
        json_object = {
            k: {
                "value": v.value,
                "translated_value": v.translated_value,
                "simplified_value": v.simplified_value,
                "ocr_boxes": [get_box_from_ocr_box(box) for box in v.ocr_boxes],
            }
            for k, v in self.legend.items()
        }
        json.dump(json_object, open(output_path, "w"), indent=4)


class GroundedArchFeat:
    def __init__(self, ocr_grounded_feats, ocr_boxes):
        self.ocr_grounded_feats = ocr_grounded_feats
        self.ocr_boxes = ocr_boxes
        self.simplified_value = None

    def add_box(self, ocr_grounded_feat, box):
        self.ocr_grounded_feats.add(ocr_grounded_feat)
        self.ocr_boxes.append(box)

    def set_simplified_value(self, simplified_value):
        self.simplified_value = simplified_value


class GroundedArcFeats:
    def __init__(self):
        self.feats = {}

    def update_arch_feat_value(self, arc_feat, ocr_box):
        stripped_arc_feat = arc_feat.strip().lower()
        if stripped_arc_feat in self.feats:
            self.feats[stripped_arc_feat].append(ocr_box)
        else:
            self.feats[stripped_arc_feat] = [ocr_box]

    def to_json(self, output_path):
        json_object = {
            k: [get_box_from_ocr_box(box) for box in v]
            for k, v in self.feats.items()
        }
        json.dump(json_object, open(output_path, "w"), indent=4)
