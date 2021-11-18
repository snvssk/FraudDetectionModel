create or replace table ml_project.data_predictions (
    timestamp timestamp,
    numOfRecords integer,
    numFraudDetected integer,
    inputDataLink string,
    processedDataLink string,
    predictionDataLink string
);



create or replace table ml_project.metric (
    timestamp timestamp,
    modelName string,
    foldNumber integer,
    testDataSetSize integer,
    trainDatasetSize integer,
    confusionMatrix struct<
    truePositive FLOAT64,
    trueNegetive FLOAT64,
    falsePositive FLOAT64,
    falseNegetive FLOAT64
    >,
    accuracy FLOAT64,
    auprc FLOAT64
);

create or replace table ml_project.model_winner (
    timestamp timestamp,
    modelName string,
    model_pkl_link string
);