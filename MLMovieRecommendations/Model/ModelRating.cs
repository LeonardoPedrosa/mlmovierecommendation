using Microsoft.ML.Data;

namespace MLMovieRecommendations.Model
{
    public class MovieRating
    {
        [LoadColumn(0)]
        public float UserId;

        [LoadColumn(1)]
        public float MovieId;

        [LoadColumn(2)]
        public float Label;
    }

}
