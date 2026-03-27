namespace LLM_App
{
    /// <summary>
    /// Splits an encoded token array into training and validation sets.
    /// </summary>
    internal interface ICorpusSplitter
    {
        /// <summary>
        /// Split <paramref name="allTokens"/> into a training set and a validation set.
        /// </summary>
        (int[] Train, int[] Validation) Split(int[] allTokens);
    }
}
