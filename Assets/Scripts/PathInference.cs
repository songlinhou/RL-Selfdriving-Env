using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.AI;

public class PathInference : MonoBehaviour
{
    private NavMeshAgent agent;
    public RoadLine roadLine;
    public Transform position;
    private LineRenderer lineRenderer;
    [SerializeField]
    private List<Transform> wayPoints;
    private List<Transform> wayPointsBackup;
    [SerializeField]
    private bool finishPathFinding;
    private bool drawComplete;
    public bool pathReady;
    // Start is called before the first frame update
    void Start()
    {
        transform.position = GameObject.FindGameObjectWithTag("Player").transform.position;
        agent = GetComponent<NavMeshAgent>();
        lineRenderer = GetComponent<LineRenderer>();
        wayPoints = new List<Transform>();
        wayPointsBackup = new List<Transform>();
        finishPathFinding = false;
        drawComplete = false;
        position = roadLine.endPoint;
        agent.SetDestination(position.position);
        pathReady = false;
        //var path = agent.path;
        //string cornerStr = "";
        //foreach (var corner in path.corners) {
        //    cornerStr += (corner + " -- ");
        //}
        //print("path corners=" + cornerStr);
    }

    // Update is called once per frame
    void Update()
    {
        //agent.SetDestination(roadLine.endPoint.position);
        
        if (!finishPathFinding)
        {
            if (Vector3.Distance(transform.position, position.position) < 3.0)
            {
                // reach the destination
                finishPathFinding = true;
            }
            GetClosestWayPoint();
        }
        else {
            drawPath();
        }
    }

    private void GetClosestWayPoint() {
        Transform nearest = roadLine.getNearestPointOf(transform);
        if (!wayPoints.Contains(nearest)) {
            wayPoints.Add(nearest);
            wayPointsBackup.Add(nearest);
        }
    }

    private void drawPath() {
        if (drawComplete) {
            return;
        }
        //lineRenderer.SetVertexCount(wayPoints.Count);
        lineRenderer.positionCount = wayPoints.Count;
        for(int i=0;i < wayPoints.Count; i ++) {
            Transform wayPoint = wayPoints[i];
            lineRenderer.SetPosition(i, wayPoint.position);
        }
        drawComplete = true;
        pathReady = true;
        print("draw complete");
    }

    public void ResetWayPoints() {
        if (wayPointsBackup.Count > 0) {
            wayPoints = new List<Transform>();
            foreach (var point in wayPointsBackup)
            {
                wayPoints.Add(point);
            }
            lineRenderer.positionCount = wayPoints.Count;
            for (int i = 0; i < wayPoints.Count; i++)
            {
                Transform wayPoint = wayPoints[i];
                lineRenderer.SetPosition(i, wayPoint.position);
            }
            print("way point reset");
        }
    }

    public Transform GetNextWayPoint() {
        if (wayPoints.Count == 0) {
            return position;
        }
        Transform t = wayPoints[0];
        return t;
    }

    public void MoveToNextWayPoint() {
        if (wayPoints.Count > 0)
        {
            wayPoints.RemoveAt(0);
            lineRenderer.positionCount = wayPoints.Count;
            for (int i = 0; i < wayPoints.Count; i++)
            {
                Transform wayPoint = wayPoints[i];
                lineRenderer.SetPosition(i, wayPoint.position);
            }
            print("waypoint updated");
        }
        else {
            print("no more way point");
        }
 
    }


}
