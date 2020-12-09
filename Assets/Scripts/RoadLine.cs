using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class RoadLine : MonoBehaviour
{
    public GameObject pointParent;
    [SerializeField]
    private List<Transform> points;
    public Transform startPoint;
    public Transform endPoint;
    private Transform playerTransform;
    public GameObject endPointMarker;
    private int direction; // this value can only be -1 or 1.
    // Start is called before the first frame update
    void Start()
    {
        playerTransform = GameObject.FindGameObjectWithTag("Player").transform;
        points = new List<Transform>(pointParent.GetComponentsInChildren<Transform>());
        print(points.Count + " points are found");
        endPointMarker.SetActive(true);
        endPointMarker.transform.position = new Vector3(endPoint.transform.position.x, 38, endPoint.transform.position.z);
        startPoint = getNearestPoint();
    }

    bool checkConsistency() {
        bool isContains = points.Contains(startPoint) && points.Contains(endPoint);
        bool isDifferent = (startPoint != endPoint);
        return isContains && isDifferent;
    }

    public Transform getNearestPoint() {
        double shortestDist = double.MaxValue;
        Transform targetTransform = null;
        foreach(Transform p in points) {
            Vector3 vec = p.position - playerTransform.position;
            double dist = Vector3.Magnitude(vec);
            if (dist < shortestDist) {
                shortestDist = dist;
                targetTransform = p;
            }
        }
        return targetTransform;
    }

    public Transform getNearestPointOf(Transform t) {
        double shortestDist = double.MaxValue;
        Transform targetTransform = null;
        foreach (Transform p in points)
        {
            if (p == t) {
                continue;
            }
            Vector3 vec = p.position - t.position;
            double dist = Vector3.Magnitude(vec);
            if (dist < shortestDist)
            {
                shortestDist = dist;
                targetTransform = p;
            }
        }
        return targetTransform;
    }
}
