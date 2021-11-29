create or replace table ml_project.data_predictions (
    timestamp timestamp,
    numOfRecords integer,
    numFraudDetected integer,
    inputDataLink string,
    processedDataLink string,
    predictionDataLink string
);

create or replace table ml_project.data_preparation (
    timestamp timestamp,
    numOfInputRecords integer,
    numOfProcessedRecords integer,
    inputDataLink string,
    processedDataLink string,
    data_prep_starttime timestamp
);

create or replace table ml_project.metric (
    timestamp timestamp,
    modelName string,
    foldNumber integer,
    testDataSetSize integer,
    trainDataSetSize integer,
    confusionMatrix struct<
    truePositive FLOAT64,
    trueNegative FLOAT64,
    falsePositive FLOAT64,
    falseNegative FLOAT64
    >,
    accuracy FLOAT64,
    auprc FLOAT64,
    fold_exec_starttime timestamp
);


create or replace table ml_project.model_winner (
    timestamp timestamp,
    modelName string,
    model_pkl_link string
);
