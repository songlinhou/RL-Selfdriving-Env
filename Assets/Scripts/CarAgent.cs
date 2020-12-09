using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using UnityEngine.UI;

public class CarAgent : Agent
{
    public Vector3 rotationEuler;
    public RoadLine roadLine;
    public int maxEpisodeSteps = 10000;
    private CarControl controller;
    private Rigidbody rigidbody;
    private float overallDist;
    private float lastDistToDest;
    EnvironmentParameters defaultParameters;
    private List<float> observation;
    [SerializeField]
    public float accScale = 1.0f;

    Vector3 destination;

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
    public PathInference pathInference;
    


    // Start is called before the first frame update
    void Start()
    {
        controller = GetComponent<CarControl>();
        rigidbody = GetComponent<Rigidbody>();
        observation = new List<float>();
        rotationEuler = gameObject.transform.rotation.eulerAngles;
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

        // the distance
        float distToDest = Vector3.Distance(transform.position, destination);
        float progressRatio = distToDest / overallDist;
        sensor.AddObservation(progressRatio);

        // path available
        if (pathInference.pathReady)
        {
            sensor.AddObservation(1.0f);
        }
        else {
            sensor.AddObservation(0.0f);

        }


        if (!addtionalFeatureNameReady) {
            additionalFeatureNames.Add("RotY");
            additionalFeatureNames.Add("RotW");
            additionalFeatureNames.Add("DistRatio");
            additionalFeatureNames.Add("PathAvailable");
        }


        print("ob size=" + sensor.ObservationSize());
        //print("count=" + ());
    }

    public void OnCarHitObstacle() {

        AddReward(-10f);
        accScale = 0.1f;
    }

    public override void OnActionReceived(float[] vectorAction)
    {
        print("OnActionReceived");
        //action space == 2
        //controller.SetAxis(vectorAction[0], vectorAction[1]);


        float xAxis = vectorAction[0];
        float yAxis = vectorAction[1];


        print("x=" + xAxis + " y=" + yAxis);

        horizontalSlider.value = xAxis;
        verticalSlider.value = yAxis; // -1 - 1
        float accelerate = controller.acclerateSpeed;
        if (yAxis <= 0)
        {
            accelerate = controller.reserseSpeed;
        }
        //float yAxis_ = (yAxis + 1f) / 2.0f;
        if (accScale > 1.0f)
        {
            accScale = 1.0f;
        }
        else
        {
            accScale += 0.005f;
        }
        transform.Translate(Vector3.forward * yAxis * accelerate * accScale * Time.deltaTime, Space.Self);
        //transform.Translate(Vector3.forward * yAxis_ * accelerate * Time.deltaTime, Space.Self);
        transform.Rotate(0.0f, xAxis * controller.rotateSpeed * Time.deltaTime, 0.0f, Space.Self);
        



        // reward function
        float distToDest = Vector3.Distance(transform.position, destination);

        Transform wayPoint = pathInference.GetNextWayPoint();

        float distToWayPoint = Vector3.Distance(transform.position, wayPoint.position);


        // reach the destination
        if (distToDest < 10.0f)
        {
            AddReward(500f);
            print("Success! Total reward=" + GetCumulativeReward());
            EndEpisode();
            return;
        }

        // reach the destination
        if (distToWayPoint < 6.0f && pathInference.pathReady)
        {
            AddReward(50f);
            print("Reach waypoint! Total reward=" + GetCumulativeReward());
            pathInference.MoveToNextWayPoint();
        }

        // distToDest
        float rewardAlongTheWay = 0;
        if (distToDest < lastDistToDest)
        {
            // reward
            rewardAlongTheWay = 0.015f;
        }
        else {
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
            //AddReward(-0.001f);
            //AddReward(-0.001f);

            //AddReward(rewardAlongTheWay);
            return;
        }

        if (StepCount >= maxEpisodeSteps) {
            EndEpisode();
        }
        
    }

    private void saveImitationLearningRecords(List<float> state, float xAxis, float yAxis) {
        string line = "";
        foreach (float var in state) {
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
        else {
            records.Add(line);
        }
        
        if (records.Count > 100) {
            actionRecorder.writeRecordsToFile(records);
            records = new List<string>();
        }
    }

    private void saveImitationLearningRecordsToCSV(List<float> state, List<string> featNames,float xAxis, float yAxis)
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
            actionRecorder.writeCSVRecordsToFile(featNames,records);
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
        for (int i = 0; i < sensorData.Count; i++) {
            observation.Add(sensorData[i]);
            if (!featureNameReady) {
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
            foreach (string feat in additionalFeatureNames) {
                featureName.Add(feat);
            }
            featureName.Add("steer");
            featureName.Add("acceleration");
        }
        //print("Heuristic obs=" + observation.Count);
        print("ROT=" + transform.rotation);
        float xAxis = Input.GetAxis("Horizontal");
        float yAxis = Input.GetAxis("Vertical");
        actionsOut[0] = xAxis;
        actionsOut[1] = yAxis;
        horizontalSlider.value = xAxis;
        verticalSlider.value = yAxis;



        //saveImitationLearningRecords(observation, xAxis, yAxis);
        saveImitationLearningRecordsToCSV(observation, featureName, xAxis, yAxis);
    }

    private void displayUI() {
        if (scoreText) {
            scoreText.text = "Score: " + GetCumulativeReward().ToString("0.00");
        }
        if (stepText) {
            stepText.text = "Steps: " + StepCount.ToString() + " / " + maxEpisodeSteps;
        }
        if (lifeText) {
            lifeText.text = "Lives: " + controller.getLife() + " / 5";
        }
        
    }


}
