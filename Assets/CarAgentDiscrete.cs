using System;
using System.Collections.Generic;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using UnityEngine;
using UnityEngine.UI;

public class CarAgentDiscrete : Agent
{
    public Vector3 rotationEuler;
    public RoadLine roadLine;
    public int maxEpisodeSteps = 10000;
    private CarControlDiscrete controller;
    private Rigidbody rigidbody;
    private float overallDist;
    private float lastDistToDest;
    EnvironmentParameters defaultParameters;
    private List<float> observation;

    Vector3 destination;
    public PathInference pathInference;
    [SerializeField]
    public float accScale = 1.0f;

    [SerializeField]
    private bool useWayPoint;


    // UI

    public Text scoreText;
    public Text stepText;
    public Text lifeText;
    public Slider horizontalSlider;
    public Slider verticalSlider;


    // IL
    private List<string> records = new List<string>(500);
    List<string> featureName = new List<string>(15);
    List<string> additionalFeatureNames = new List<string>(5);
    public ActionRecorder actionRecorder;
    private readonly float[,] actionVector = new float[5, 5]
        {
            {1,2,3,4,5},
            {6,7,8,9,10},
            {11,12,13,14,15},
            {16,17,18,19,20},
            {21,22,23,24,25},
        };



    // Start is called before the first frame update
    void Start()
    {
        controller = GetComponent<CarControlDiscrete>();
        rigidbody = GetComponent<Rigidbody>();
        observation = new List<float>();

    }

    private void FixedUpdate()
    {
        //RequestAction();
        //RequestDecision();
    }

    public override void Initialize()
    {
        base.Initialize();
        print("Initialize");
        Vector3 endPos = roadLine.endPoint.position;
        destination = new Vector3(endPos.x, 0f, endPos.z);

        Vector3 startPos = roadLine.startPoint.position;
        Vector3 initPos = new Vector3(startPos.x, 0f, startPos.z);
        overallDist = Vector3.Distance(initPos, endPos);
        defaultParameters = Academy.Instance.EnvironmentParameters;
    }


    public override void OnEpisodeBegin()
    {
        // reset position
        print("OnEpisodeBegin");
        Vector3 startPos = roadLine.startPoint.position;
        transform.position = new Vector3(startPos.x, 0f, startPos.z);
        transform.rotation = Quaternion.Euler(rotationEuler);
        rigidbody.velocity = Vector3.zero;
        rigidbody.angularVelocity = Vector3.zero;
        controller.resetCarStatus();
        lastDistToDest = overallDist;
        pathInference.ResetWayPoints();
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        //base.CollectObservations(sensor);
        print("CollectObservations");
        //List<float> sensorData = controller.GetSensorData();
        //foreach (float d in sensorData) {
        //sensor.AddObservation(d);
        //print(d);
        //}

        //sensor.AddObservation(rigidbody.velocity);
        //print("rigidbody.velocity=" + rigidbody.velocity);
        bool addtionalFeatureNameReady = (additionalFeatureNames.Count != 0);
        // direction
        Quaternion rot = transform.rotation;
        sensor.AddObservation(rot.y);
        sensor.AddObservation(rot.w);

        float distToDest = Vector3.Distance(transform.position, destination);
        float progressRatio = distToDest / overallDist;
        sensor.AddObservation(progressRatio);

        // path available
        if (pathInference.pathReady)
        {
            sensor.AddObservation(1.0f);
        }
        else
        {
            sensor.AddObservation(0.0f);

        }

        if (!addtionalFeatureNameReady)
        {
            additionalFeatureNames.Add("RotY");
            additionalFeatureNames.Add("RotW");
            additionalFeatureNames.Add("DistRatio");
            additionalFeatureNames.Add("PathAvailable");
        }

        print("ob size=" + sensor.ObservationSize());
        //print("count=" + ());
    }

    public void OnCarHitObstacle()
    {
        AddReward(-10f);
        accScale = 0.1f;
    }

    public override void OnActionReceived(float[] vectorAction)
    {
        print("OnActionReceived");
        //action space == 2
        //controller.SetAxis(vectorAction[0], vectorAction[1]);
        float xAxis = 0;
        float yAxis = 0;

        for (int x = 0; x < 5; ++x)
        {
            for (int y = 0; y < 5; ++y)
            {
                if (actionVector[x, y].Equals(vectorAction[0]))
                {
                    xAxis = (float)(x - 2) / 2;
                    yAxis = (float)(y - 2) / 2;
                }
            }
        }
        print("x=" + xAxis + " y=" + yAxis);

        horizontalSlider.value = xAxis;
        verticalSlider.value = yAxis;
        float accelerate = controller.acclerateSpeed;
        if (yAxis <= 0)
        {
            accelerate = controller.reserseSpeed;
        }
        if (accScale > 1.0f)
        {
            accScale = 1.0f;
        }
        else
        {
            accScale += 0.005f;
        }
        transform.Translate(Vector3.forward * yAxis * accelerate * accScale * Time.deltaTime, Space.Self);
        transform.Rotate(0.0f, xAxis * controller.rotateSpeed * Time.deltaTime, 0.0f, Space.Self);



        // reward function
        float distToDest = Vector3.Distance(transform.position, destination);

        Transform wayPoint = pathInference.GetNextWayPoint();

        float distToWayPoint = Vector3.Distance(transform.position, wayPoint.position);


        // reach the destination
        if (distToDest < 3.0f)
        {
            AddReward(500f);
            print("Success! Total reward=" + GetCumulativeReward());
            EndEpisode();
            return;
        }

        // reach the waypoint
        if (useWayPoint) {
            if (distToWayPoint < 6.0f && pathInference.pathReady)
            {
                AddReward(50f);
                print("Reach waypoint! Total reward=" + GetCumulativeReward());
                pathInference.MoveToNextWayPoint();
            }
        }
        

        // distToDest
        float rewardAlongTheWay = 0;
        if (distToDest < lastDistToDest)
        {
            // reward
            rewardAlongTheWay = 0.015f;
        }
        else
        {
            rewardAlongTheWay = -0.015f;
        }

        lastDistToDest = distToDest;

        //float rewardAlongTheWay = (float)(1.0 * ((overallDist - distToDest) / overallDist));

        //print("step count=" + StepCount + " Reward=" + GetCumulativeReward());
        displayUI();
        // life become 0
        if (controller.isDead())
        {
            AddReward(-50f);
            EndEpisode();
            return;
        }
        else if (StepCount < maxEpisodeSteps)
        {
            if (!useWayPoint) {
                AddReward(-0.001f);
                AddReward(rewardAlongTheWay);
            }
            
            return;
        }

        if (StepCount >= maxEpisodeSteps)
        {
            EndEpisode();
        }

    }

    private void saveImitationLearningRecords(List<float> state, float xAxis, float yAxis)
    {
        string line = "";
        foreach (float var in state)
        {
            line += (var + ",");
        }
        line = line.Substring(0, line.Length - 1); // remove the last ,
        line += ("#" + xAxis + "," + yAxis);
        print("line=" + line);
        if (records.Count > 0)
        {
            if (!records[records.Count - 1].Equals(line))
            {
                records.Add(line);
            }
        }
        else
        {
            records.Add(line);
        }

        if (records.Count > 100)
        {
            actionRecorder.writeRecordsToFile(records);
            records = new List<string>();
        }
    }

    private void saveImitationLearningRecordsToCSV(List<float> state, List<string> featNames, float actionType)
    {
        string line = "";
        foreach (float var in state)
        {
            line += (var + ",");
        }
        int action = ((int)actionType);
        line += (action);
        print("line=" + line);
        if (records.Count > 0)
        {
            if (!records[records.Count - 1].Equals(line))
            {
                records.Add(line);
            }
        }
        else
        {
            records.Add(line);
        }

        if (records.Count > 100)
        {
            actionRecorder.writeCSVRecordsToFile(featNames, records);
            records = new List<string>();
        }
    }

    private void saveImitationLearningRecordsToCSV(List<float> state, List<string> featNames, float xAxis, float yAxis)
    {
        string line = "";
        foreach (float var in state)
        {
            line += (var + ",");
        }
        line += (xAxis + "," + yAxis);
        print("line=" + line);
        if (records.Count > 0)
        {
            if (!records[records.Count - 1].Equals(line))
            {
                records.Add(line);
            }
        }
        else
        {
            records.Add(line);
        }

        if (records.Count > 100)
        {
            actionRecorder.writeCSVRecordsToFile(featNames, records);
            records = new List<string>();
        }
    }

    public override void Heuristic(float[] actionsOut)
    {
        base.Heuristic(actionsOut);
        bool featureNameReady = (featureName.Count != 0);

        var obs = GetObservations();
        List<float> sensorData = controller.GetSensorData();
        observation = new List<float>();
        for (int i = 0; i < sensorData.Count; i++)
        {
            observation.Add(sensorData[i]);
            if (!featureNameReady)
            {
                featureName.Add("SD" + i);
            }
        }

        //foreach (var ob in sensorData)
        //{
        //    observation.Add(ob);
        //}
        foreach (var ob in obs)
        {
            print("obs=" + ob);
            observation.Add(ob);
        }
        if (!featureNameReady)
        {
            foreach (string feat in additionalFeatureNames)
            {
                featureName.Add(feat);
            }
            featureName.Add("action");
        }
        //print("Heuristic obs=" + observation.Count);
        print("ROT=" + transform.rotation);
        float xAxis = Input.GetAxis("Horizontal");
        float yAxis = Input.GetAxis("Vertical");
        int discretexAxis = (int)Mathf.Round(((xAxis * 10) % 2));
        int discreteyAxis = (int)Mathf.Round(((yAxis * 10) % 2));

        actionsOut[0] = actionVector[discretexAxis + 2, discreteyAxis + 2];

        horizontalSlider.value = xAxis;
        verticalSlider.value = yAxis;

        //saveImitationLearningRecords(observation, xAxis, yAxis);
        saveImitationLearningRecordsToCSV(observation, featureName, actionsOut[0]);
    }

    private void displayUI()
    {
        if (scoreText)
        {
            scoreText.text = "Score: " + GetCumulativeReward().ToString("0.00");
        }
        if (stepText)
        {
            stepText.text = "Steps: " + StepCount.ToString() + " / " + maxEpisodeSteps;
        }
        if (lifeText)
        {
            lifeText.text = "Lives: " + controller.getLife() + " / 5";
        }

    }
}