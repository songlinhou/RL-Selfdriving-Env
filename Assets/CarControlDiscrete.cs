using System.Collections;
using System.Collections.Generic;
using Unity.MLAgents.Sensors;
using UnityEngine;

public class CarControlDiscrete : MonoBehaviour
{
    public GameObject player;
    public float rotateSpeed = 20f;
    public float acclerateSpeed = 20f;
    public float reserseSpeed = 5f; // 5
    [SerializeField]
    private int lives;
    public RoadLine roadLine;
    private RayPerceptionSensorComponent3D distanceSensorComponent;
    private RayPerceptionSensor sensor;
    private CarAgentDiscrete agent;

    public float xAxis = 0f;
    public float yAxis = 0f;
    public bool mannuel = false;


    // Start is called before the first frame update
    void Start()
    {
        distanceSensorComponent = GetComponent<RayPerceptionSensorComponent3D>();
        // sensor = (RayPerceptionSensor)distanceSensorComponent.CreateSensor();
        agent = GetComponent<CarAgentDiscrete>();
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

    public void SetAxis(float xAxis, float yAxis)
    {
        this.xAxis = Mathf.Round(((xAxis * 10) % 2)) / 2;
        this.yAxis = Mathf.Round(((yAxis * 10) % 2)) / 2;
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

    public bool isDead()
    {
        return lives <= 0;
    }

    public void resetCarStatus()
    {
        lives = 5;
    }

    public int getLife()
    {
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
            print("life=" + lives);
        }
        else
        {
            print("non obstacle=" + collision.gameObject.name);
        }
    }
}
