# Winner-Predictor-in-League-of-Legends-Reworked

Predict the win possibilities of the team in League of Legends based on their team combination. It uses custom and personal trained BERT model for word embedding and uses the Multilayer Perceptrons (MLPs) model to predict the winner/win possibilities using the embedded data from BERT.

Tested several model for classification:

    MLP: 51.2%
    CNN: 54.1%

Results: Since the game can be end in one mistake and other user's skill or luck, the combination of the team might affect the game result, but cannot make significant effect on winning. Therefore, I might work this algorithm for soccer game which the player and position has significant impact on match.
