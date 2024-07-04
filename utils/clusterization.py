from sklearn.cluster import OPTICS, DBSCAN

SCALE = 0.065

def clusterize(det, img_w):
    if len(det) < 3:
        return det

    eps = img_w * SCALE
    l = [[x.item(), y.item()] for x, y in det[:, :2]]
    clusterLabels = DBSCAN(eps=eps, min_samples=3).fit_predict(l)

    clusters = {}
    retVal = []
    for index, label in enumerate(clusterLabels):
        rect = det[index, :4]
        
        if label == -1:
            retVal.append([*rect, *det[index, 4:6]])
            continue

        if label in clusters:
            cluster = clusters[label]
            cluster[0] = min(cluster[0], rect[0]) # left
            cluster[1] = min(cluster[1], rect[1]) # top
            cluster[2] = max(cluster[2], rect[2]) # right
            cluster[3] = max(cluster[3], rect[3]) # bottom
        else:
            clusters[label] = rect
    
        for label in clusters:
            retVal.append([*clusters[label], *det[index, 4:6]])
    
    return retVal