import pygame
import sys
from board import Board
from agent import Agent
import pickle
import math

class TicTacToeGame:
    def __init__(self):
        pygame.init()
        
        self.WIDTH = 600
        self.HEIGHT = 700
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        pygame.display.set_caption("Tic-Tac-Toe vs Trained Agent")
        
        self.CELL_SIZE = 150
        self.MARGIN = 50
        self.BOARD_START_Y = 100
        
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.BLUE = (0, 100, 255)
        self.RED = (255, 0, 0)
        self.GRAY = (200, 200, 200)
        self.GREEN = (0, 150, 0)
        self.ORANGE = (255, 165, 0)
        
        self.font_title = pygame.font.Font(None, 36)
        self.font_text = pygame.font.Font(None, 28)
        self.font_large = pygame.font.Font(None, 80)
        
        self.board = Board()
        self.agent = self.load_agent()
        self.state = tuple('.' * 9)
        self.game_over = False
        self.status_text = "Your turn (X)"
        self.is_agent_thinking = False
        self.clock = pygame.time.Clock()
    
    def load_agent(self):
        """Load the trained agent from file"""
        try:
            with open('trained_agent.pkl', 'rb') as f:
                agent = pickle.load(f)
            agent.epsilon = 0
            return agent
        except FileNotFoundError:
            print("Error: trained_agent.pkl not found.")
            print("Please run 'python train.py' first to train the agent.")
            pygame.quit()
            sys.exit()
    
    def get_cell_from_mouse(self, pos):
        """Convert mouse position to board cell"""
        x, y = pos
        
        # Check if within board bounds
        if x < self.MARGIN or x > self.MARGIN + self.CELL_SIZE * 3:
            return None
        if y < self.BOARD_START_Y or y > self.BOARD_START_Y + self.CELL_SIZE * 3:
            return None
        
        col = (x - self.MARGIN) // self.CELL_SIZE
        row = (y - self.BOARD_START_Y) // self.CELL_SIZE
        
        return row * 3 + col
    
    def draw_board(self):
        """Draw the game board"""
        self.screen.fill(self.WHITE)
        
        # Title
        title = self.font_title.render("Tic-Tac-Toe vs Trained Agent", True, self.BLACK)
        self.screen.blit(title, (self.WIDTH // 2 - title.get_width() // 2, 20))
        
        subtitle = self.font_text.render("You are X (blue), Agent is O (red)", True, self.GRAY)
        self.screen.blit(subtitle, (self.WIDTH // 2 - subtitle.get_width() // 2, 55))
        
        # Draw grid
        for row in range(4):
            y = self.BOARD_START_Y + row * self.CELL_SIZE
            pygame.draw.line(self.screen, self.BLACK, 
                           (self.MARGIN, y), 
                           (self.MARGIN + self.CELL_SIZE * 3, y), 2)
        
        for col in range(4):
            x = self.MARGIN + col * self.CELL_SIZE
            pygame.draw.line(self.screen, self.BLACK,
                           (x, self.BOARD_START_Y),
                           (x, self.BOARD_START_Y + self.CELL_SIZE * 3), 2)
        
        # Draw X's and O's
        for i in range(9):
            row = i // 3
            col = i % 3
            x = self.MARGIN + col * self.CELL_SIZE + self.CELL_SIZE // 2
            y = self.BOARD_START_Y + row * self.CELL_SIZE + self.CELL_SIZE // 2
            
            if self.state[i] == 'X':
                text = self.font_large.render('X', True, self.BLUE)
                self.screen.blit(text, (x - text.get_width() // 2, y - text.get_height() // 2))
            elif self.state[i] == 'O':
                text = self.font_large.render('O', True, self.RED)
                self.screen.blit(text, (x - text.get_width() // 2, y - text.get_height() // 2))
        
        # Draw status
        status_color = self.BLACK
        if "won" in self.status_text:
            status_color = self.GREEN if "You" in self.status_text else self.RED
        elif "draw" in self.status_text:
            status_color = self.ORANGE
        
        status = self.font_text.render(self.status_text, True, status_color)
        self.screen.blit(status, (self.WIDTH // 2 - status.get_width() // 2, self.BOARD_START_Y + self.CELL_SIZE * 3 + 40))
        
        # Draw buttons
        self.draw_button("New Game", 150, 630)
        if self.is_agent_thinking:
            thinking = self.font_text.render("Agent thinking...", True, self.GRAY)
            self.screen.blit(thinking, (self.WIDTH // 2 - thinking.get_width() // 2, 560))
        
        pygame.display.flip()
    
    def draw_button(self, text, x, y, width=150, height=40):
        """Draw a button"""
        pygame.draw.rect(self.screen, self.GREEN, (x - width // 2, y - height // 2, width, height))
        pygame.draw.rect(self.screen, self.BLACK, (x - width // 2, y - height // 2, width, height), 2)
        
        label = self.font_text.render(text, True, self.WHITE)
        self.screen.blit(label, (x - label.get_width() // 2, y - label.get_height() // 2))
        
        return pygame.Rect(x - width // 2, y - height // 2, width, height)
    
    def player_move(self, position):
        """Handle player's move"""
        if self.game_over or self.is_agent_thinking:
            return
        
        legal_moves = self.board.get_legal_moves(self.state)
        if position not in legal_moves:
            return
        
        # Player move (X)
        self.state = self.board.make_a_move(self.state, position, 'X')
        
        # Check if player won
        winner = self.board.check_winner(self.state)
        if winner == 'X':
            self.status_text = "You won!"
            self.game_over = True
            return
        
        # Check for draw
        if self.board.check_draw(self.state):
            self.status_text = "It's a draw!"
            self.game_over = True
            return
        
        # Agent's turn
        self.is_agent_thinking = True
    
    def agent_move(self):
        """Handle agent's move"""
        legal_moves = self.board.get_legal_moves(self.state)
        if not legal_moves:
            self.status_text = "It's a draw!"
            self.game_over = True
            self.is_agent_thinking = False
            return
        
        action = self.agent.choose_action(self.state, legal_moves, self.board)
        self.state = self.board.make_a_move(self.state, action, 'O')
        
        # Check if agent won
        winner = self.board.check_winner(self.state)
        if winner == 'O':
            self.status_text = "Agent won!"
            self.game_over = True
            self.is_agent_thinking = False
            return
        
        # Check for draw
        if self.board.check_draw(self.state):
            self.status_text = "It's a draw!"
            self.game_over = True
            self.is_agent_thinking = False
            return
        
        # Back to player's turn
        self.status_text = "Your turn (X)"
        self.is_agent_thinking = False
    
    def reset_game(self):
        """Start a new game"""
        self.state = tuple('.' * 9)
        self.game_over = False
        self.status_text = "Your turn (X)"
        self.is_agent_thinking = False
    
    def run(self):
        """Main game loop"""
        running = True
        agent_delay = 0
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    pos = event.pos
                    
                    # Check if New Game button clicked
                    if pos[1] > 610:
                        self.reset_game()
                    else:
                        # Check if board clicked
                        cell = self.get_cell_from_mouse(pos)
                        if cell is not None:
                            self.player_move(cell)
            
            # Handle agent's delayed move
            if self.is_agent_thinking:
                agent_delay += 1
                if agent_delay > 30:  # ~0.5 second delay at 60 FPS
                    self.agent_move()
                    agent_delay = 0
            
            self.draw_board()
            self.clock.tick(60)
        
        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    game = TicTacToeGame()
    game.run()
