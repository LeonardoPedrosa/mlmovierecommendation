using Microsoft.ML.Data;

namespace MLMovieRecommendations.Model
{
    public class MovieRatingPrediction
    {
        [ColumnName("Score")]
        public float Score;

        // You can add additional properties if needed
        // For example, you might want to include MovieId in the prediction result
        [ColumnName("MovieId")]
        public float MovieId;
    }
}
