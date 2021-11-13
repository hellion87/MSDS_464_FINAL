from .agent import Agent, StockfishAgent
from .chess_env import ChessEnvV1


class EloEvaluator:
    def __init__(self, agent: Agent, n: int):
        self.agent = agent
        self.n = n

    def _play_game(self, opponent_elo) -> int:
        opponent = StockfishAgent(name='Stockfish', stockfish_path='stockfish', elo=opponent_elo)
        env = ChessEnvV1(opponent=opponent)
        obs = env.reset()
        white_move = self.agent.observe(0, obs)
        obs, reward, is_done = env.step(white_move)
        while not is_done:
            white_move = self.agent.observe(0, obs)
            obs, reward, is_done = env.step(white_move)
        return reward

    @staticmethod
    def _calc_elo(total_opp_rating: int, wins: int, losses: int, games_played: int) -> int:
        elo = int((total_opp_rating + 400 * (wins - losses)) / games_played)
        # minimum ELO score is 100
        elo = max(elo, 100)
        return elo

    def eval(self) -> tuple:
        # initialize elo guess at 1500
        elo = 1500
        wins = 0
        losses = 0
        total_opp_rating = 0
        games_played = 0
        for n in range(self.n):
            # play opponent at current skill level
            # opponent can only play between 1350 and 2850
            opp_elo = min(max(elo, 1350), 2850)
            total_opp_rating += opp_elo
            print('playing game number', n + 1, 'against opponent with ELO score of', opp_elo)
            result = self._play_game(opp_elo)
            games_played += 1
            if result == 1:
                wins += 1
                print('won game!')
            elif result == -1:
                losses += 1
                print('lost game :(')
            elif result == 0:
                print('draw  ¯\_(ツ)_/¯')
            else:
                raise Exception('Unknown game result')
            elo = self._calc_elo(total_opp_rating, wins, losses, games_played)
            print('current ELO estimate:', elo)

        info = {'wins': wins, 'losses': losses, 'games_played': games_played}
        return elo, info
