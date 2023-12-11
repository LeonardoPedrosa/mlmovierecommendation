using MLMovieRecommendations.Model;
using Microsoft.ML.Trainers;
using Microsoft.ML;

Console.WriteLine("Hello, World!");

var context = new MLContext();

var trainingDataLocation = "C:\\Users\\leonardo.pedrosa\\source\\repos\\MLMovieRecommendation\\MLMovieRecommendations\\MLMovieRecommendations\\Data\\DataMovieRecommendation.csv";
// Load data

IDataView trainingDataView = context.Data.LoadFromTextFile<MovieRating>(trainingDataLocation, hasHeader: true, separatorChar: ',');

var data = context.Transforms.Conversion.MapValueToKey(outputColumnName: "userIdEncoded", inputColumnName: nameof(MovieRating.UserId))
    .Append(context.Transforms.Conversion.MapValueToKey(outputColumnName: "movieIdEncoded", inputColumnName: nameof(MovieRating.MovieId)));

// Define the model
var recommenderOptions = new MatrixFactorizationTrainer.Options
{
    MatrixColumnIndexColumnName = "userIdEncoded",
    MatrixRowIndexColumnName = "movieIdEncoded",
    LabelColumnName = "Label",
    NumberOfIterations = 20,
    ApproximationRank = 100
};

// Train the model
var trainingPipeLine = data.Append(context.Recommendation().Trainers.MatrixFactorization(recommenderOptions));
//STEP 5: Train the model fitting to the DataSet

Console.WriteLine("=============== Training the model ===============");
ITransformer model = trainingPipeLine.Fit(trainingDataView);

// Evaluate the model (optional)
Console.WriteLine("=============== Evaluating the model ===============");
IDataView testDataView = context.Data.LoadFromTextFile<MovieRating>(trainingDataLocation, hasHeader: true, separatorChar: ',');
var prediction = model.Transform(testDataView);
var metrics = context.Regression.Evaluate(prediction, labelColumnName: "Label", scoreColumnName: "Score");
Console.WriteLine("The model evaluation metrics RootMeanSquaredError:" + metrics.RootMeanSquaredError);


// Get user input
Console.Write("Enter user id for movie recommendations: ");
var userId = float.Parse(Console.ReadLine());

// Create user movie ratings
var userRatings = new List<MovieRating>();
Console.WriteLine("Enter movie ratings (MovieId Rating), type 'done' to finish:");

while (true)
{
    var input = Console.ReadLine();
    if (input.ToLower() == "done") break;

    var parts = input.Split(' ');
    var movieId = float.Parse(parts[0]);
    var rating = float.Parse(parts[1]);

    userRatings.Add(new MovieRating { UserId = userId, MovieId = movieId, Label = rating });
}

// Make recommendations
var userRatingData = context.Data.LoadFromEnumerable(userRatings);
var userPredictions = model.Transform(userRatingData);

// Display top movie recommendations
var topRecommendations = context.Data.CreateEnumerable<MovieRatingPrediction>(userPredictions, reuseRowObject: false)
    .OrderByDescending(r => r.Score)
    .Take(5);

Console.WriteLine("Top movie recommendations:");
foreach (var recommendation in topRecommendations)
{
    Console.WriteLine($"MovieId: {recommendation.MovieId}, Score: {recommendation.Score}");
}