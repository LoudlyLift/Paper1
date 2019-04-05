import config
import equipment
import qlearning
import qtable

w = config.newWorld()

ql = qlearning.qlearning(env=w,
                         compute_randact=lambda episodeNum: 0.1,
                         consPlayer=qtable.qtable,
                         player_config=config.qtableConfig,
                         future_discount=config.future_discount)

ql.runEpisodes(10)
