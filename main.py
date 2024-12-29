import neat
import pygame
import os
import pickle
import matplotlib.pyplot as plt
from pong import Game
import numpy as np

class PongGame:
    def __init__(self, screen, width, height):
        self.game = Game(screen, width, height)
        self.leftPaddle = self.game.left_paddle
        self.rightPaddle = self.game.right_paddle
        self.ball = self.game.ball

    def testAI(self, trainedGenome, config):
        net = neat.nn.FeedForwardNetwork.create(trainedGenome, config)
        clock = pygame.time.Clock()
        running = True

        while running:
            clock.tick(60)
            handleEvent()

            keys = pygame.key.get_pressed()
            if keys[pygame.K_z]:
                self.game.move_paddle(True, True)  # Left Paddle Up
            if keys[pygame.K_s]:
                self.game.move_paddle(True, False)  # Left Paddle Down

            output = net.activate((self.rightPaddle.y, self.ball.y, abs(self.rightPaddle.x - self.ball.x)))
            decision = output.index(max(output))

            if decision == 1:
                self.game.move_paddle(False, True)  # Right Paddle Up
            elif decision == 2:
                self.game.move_paddle(False, False)  # Right Paddle Down

            self.game.loop()
            self.game.draw(True, False)
            pygame.display.update()

    def trainAI(self, leftGenome, rightGenome, config):
        leftNet = neat.nn.FeedForwardNetwork.create(leftGenome, config)
        rightNet = neat.nn.FeedForwardNetwork.create(rightGenome, config)
        running = True

        while running:
            handleEvent()
            
            leftDecision = decideMovement(leftNet, self.leftPaddle, self.ball)
            rightDecision = decideMovement(rightNet, self.rightPaddle, self.ball)

            if leftDecision == 1:
                self.game.move_paddle(True, True)  # Left Paddle Up
            elif leftDecision == 2:
                self.game.move_paddle(True, False)  # Left Paddle Down
                
            if rightDecision == 1:
                self.game.move_paddle(False, True)  # Right Paddle Up
            elif rightDecision == 2:
                self.game.move_paddle(False, False)  # Right Paddle Down

            gameInfo = self.game.loop()
            self.game.draw(False, True)
            pygame.display.update()

            if gameInfo.left_score >= 1 or gameInfo.right_score >= 1 or gameInfo.left_hits > 50:
                self.calculateFitness(leftGenome, rightGenome, gameInfo)
                break

    @staticmethod
    def calculateFitness(leftGenome, rightGenome, gameInfo):
        leftGenome.fitness += gameInfo.left_hits
        rightGenome.fitness += gameInfo.right_hits

def decideMovement(network, paddle, ball):
    output = network.activate((paddle.y, ball.y, abs(paddle.x - ball.x)))
    decision = output.index(max(output))
    return decision
    
    
def handleEvent():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()
def evaluateGenomes(genomes, config):
    width, height = 700, 500
    screen = pygame.display.set_mode((width, height))

    for i, (_, leftGenome) in enumerate(genomes):
        if i == len(genomes) - 1:
            break

        leftGenome.fitness = 0
        for _, rightGenome in genomes[i + 1:]:
            if rightGenome.fitness is None:
                rightGenome.fitness = 0

            game = PongGame(screen, width, height)
            game.trainAI(leftGenome, rightGenome, config)


def moving_average(data, window_size):
    """Return the moving average of the function"""
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

def plotStatistics(stats, window_size=5):
    generations = range(len(stats.most_fit_genomes))
    maxFitness = [genome.fitness for genome in stats.most_fit_genomes]
    avgFitness = stats.get_fitness_mean()

    smoothed_maxFitness = moving_average(maxFitness, window_size)
    smoothed_generations = range(window_size - 1, len(generations))

    plt.figure(figsize=(10, 6))
    plt.plot(generations, maxFitness, label="Max Fitness", alpha=0.7)
    plt.plot(smoothed_generations, smoothed_maxFitness, label=f"Smoothed Max Fitness (window={window_size})", linestyle="--", color="orange")
    plt.plot(generations, avgFitness, label="Avg Fitness", alpha=0.7)
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.title("Evolution of Fitness Over Generations")
    plt.legend()
    plt.grid()
    plt.show()


def runNEAT(configPath):
    pop = neat.Population(configPath)
    pop.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(neat.Checkpointer(1))

    winner = pop.run(evaluateGenomes, 30)  # Take the genome the fit the criteria in config or take the best genome out of 30 generations

    with open("best.pickle", "wb") as f:
        pickle.dump(winner, f)

    with open("stats.pkl", "wb") as f:
        pickle.dump(stats, f)

    plotStatistics(stats)


def testAI(configPath):
    with open("best.pickle", "rb") as f:
        bestGenome = pickle.load(f)

    width, height = 700, 500
    screen = pygame.display.set_mode((width, height))
    game = PongGame(screen, width, height)
    game.testAI(bestGenome, configPath)


if __name__ == "__main__":
    localDir = os.path.dirname(__file__)
    configPath = os.path.join(localDir, "config.txt")
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, configPath)
    
    with open("stats.pkl", "rb") as f:
        stats = pickle.load(f)

    # Uncomment one of these based on your use case:
    
    # runNEAT(config)
    plotStatistics(stats,15)
    testAI(config)
    
