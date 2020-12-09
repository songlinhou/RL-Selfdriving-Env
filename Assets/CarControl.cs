using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents.Sensors;
using Unity.MLAgents;

public class CarControl : MonoBehaviour
{
    public GameObject player;
    public float rotateSpeed = 20f;
    public float acclerateSpeed = 20f;
    public float reserseSpeed = 0f; // 5
    [SerializeField]
    private int lives;
    public RoadLine roadLine;
    private RayPerceptionSensorComponent3D distanceSensorComponent;
    private RayPerceptionSensor sensor;
    private CarAgent agent;

    public float xAxis = 0f;
    public float yAxis = 0f;
    public bool mannuel = false;


    // Start is called before the first frame update
    void Start()
    {
        distanceSensorComponent = GetComponent<RayPerceptionSensorComponent3D>();
        // sensor = (RayPerceptionSensor)distanceSensorComponent.CreateSensor();
        agent = GetComponent<CarAgent>();
        resetCarStatus();
    }


    // Update is called once per frame
    void Update()
    {
        //if (mannuel) {
        //    xAxis = Input.GetAxis("Horizontal");
        //    yAxis = Input.GetAxis("Vertical");
        //}
        

        //transform.Translate(Vector3.forward * yAxis * acclerateSpeed * Time.deltaTime, Space.Self);
        //transform.Rotate(0.0f, xAxis * rotateSpeed * Time.deltaTime, 0.0f, Space.Self);
    }

    public void SetAxis(float xAxis, float yAxis) {
        this.xAxis = xAxis;
        this.yAxis = yAxis;
    }

    public List<float> GetSensorData()
    {
        // check for the sensor data
        RayPerceptionInput input = distanceSensorComponent.GetRayPerceptionInput();
        RayPerceptionOutput output = RayPerceptionSensor.Perceive(input);
        var list = output.RayOutputs;
        //RayPerceptionOutput.RayOutput output0 = (RayPerceptionOutput.RayOutput)list.GetValue(0);
        //print(output0.HitFraction);
        List<float> hitFractions = new List<float>();
        foreach (var listItem in list)
        {
            RayPerceptionOutput.RayOutput outputRef = (RayPerceptionOutput.RayOutput)listItem;
            hitFractions.Add(outputRef.HitFraction);
        }
        return hitFractions;
    }

    public bool isDead() {
        return lives <= 0;
    }

    public void resetCarStatus() {
        lives = 5;
    }

    public int getLife() {
        return lives;
    }

    private void OnCollisionEnter(Collision collision)
    {
        if (collision.gameObject.layer == 8) // obstacle
        {
            print("find obstacle");
            print("object name=" + collision.gameObject.name);
            //lives -= 1;
            Transform nearest = roadLine.getNearestPoint();
            this.transform.position = new Vector3(nearest.position.x, 0, nearest.position.z);
            gameObject.GetComponent<Rigidbody>().velocity = Vector3.zero;
            agent.OnCarHitObstacle();
            //Transform secondNearest = roadLine.getNearestPointOf(nearest);
            //Vector3 roadDirection = secondNearest.position - nearest.position;
            //Vector3 destDirection = roadLine.endPoint.position - nearest.position;
            //float angle = Vector3.Angle(destDirection.normalized ,roadDirection.normalized);
            //Vector3 lookAtVec = new Vector3(secondNearest.position.x, 0, secondNearest.position.z);

            //print("NEAR=" + nearest.name + " 2NEAR=" + secondNearest.name);

            //float dir = 1;
            //if (angle > 90) {
            //    lookAtVec *= -1;
            //    dir = -1;
            //}

            //transform.LookAt(lookAtVec);
            //float y = transform.rotation.eulerAngles.y;
            //if (y < 0) {
            //    y += 360;
            //}
            //if (y < 45)
            //{
            //    y = 0;
            //}
            //else if (y < 135) {
            //    y = 90;
            //}
            //else if (y < 225)
            //{
            //    y = 180;
            //}
            //else if (y < 315)
            //{
            //    y = 270;
            //}
            //print("y=" + y);
            ////transform.rotation = Quaternion.Euler(roadDirection.normalized * dir);
            //// transform.rotation.SetEulerAngles(Vector3.up * y);

            //gameObject.transform.eulerAngles = new Vector3(0, y* dir, 0);
            print("life=" + lives);
        }
        else {
            print("non obstacle=" + collision.gameObject.name);
        }
    }
}
