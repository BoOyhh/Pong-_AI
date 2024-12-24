import pygame
import random

pygame.init()


WIDTH,HEIGHT = 800,600

screen = pygame.display.set_mode((WIDTH,HEIGHT))

WHITE = (255,255,255)
BLACK  = (0,0,0)

ball = pygame.Rect(WIDTH // 2 -15,HEIGHT//2 -15 ,30,30)
player=  pygame.Rect(WIDTH-20, HEIGHT//2 -70, 10,140)
opponent = pygame.Rect(10,HEIGHT//2 -70,10,140)
ball_speed_x = 7 * random.choice((1, -1))
ball_speed_y = 7 * random.choice((1, -1))
player_speed = 0
opponent_speed = 7

clock = pygame.time.Clock()

running = True

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:
                player_speed = -7
            if event.key == pygame.K_DOWN:
                player_speed = 7
        if event.type == pygame.KEYUP:
            if event.key in (pygame.K_UP, pygame.K_DOWN):
                player_speed = 0

    ball.x += ball_speed_x
    ball.y += ball_speed_y
    player.y += player_speed

    if opponent.top < ball.y:
        opponent.y += opponent_speed
    if opponent.bottom > ball.y:
        opponent.y -= opponent_speed

    player.clamp_ip(screen.get_rect())
    opponent.clamp_ip(screen.get_rect())

    if ball.top <= 0 or ball.bottom >= HEIGHT:
        ball_speed_y *= -1
    if ball.colliderect(player) or ball.colliderect(opponent):
        ball_speed_x *= -1

    if ball.left <= 0 or ball.right >= WIDTH:
        ball.center = (WIDTH // 2, HEIGHT // 2)
        ball_speed_x *= random.choice((1, -1))
        ball_speed_y *= random.choice((1, -1))

    screen.fill(BLACK)
    pygame.draw.rect(screen, WHITE, player)
    pygame.draw.rect(screen, WHITE, opponent)
    pygame.draw.ellipse(screen, WHITE, ball)
    pygame.draw.aaline(screen, WHITE, (WIDTH // 2, 0), (WIDTH // 2, HEIGHT))

    pygame.display.flip()
    clock.tick(60)

pygame.quit()