def normalize_bbox(bbox, width, height):
    """
    원본 좌표(x1, y1, x2, y2)를 0~1000 사이 정수로 변환합니다.
    LayoutLMv3는 0~1000 스케일의 좌표만 이해합니다.
    """
    return [
        int(1000 * (bbox[0] / width)),
        int(1000 * (bbox[1] / height)),
        int(1000 * (bbox[2] / width)),
        int(1000 * (bbox[3] / height)),
    ]
