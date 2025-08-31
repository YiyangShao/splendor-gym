"""
Demonstrate the simplified Take 3 logic proposed by the user.
"""

from splendor_gym.engine import initial_state
from splendor_gym.engine.encode import TAKE3_OFFSET, TAKE3_COMBOS, TAKE3_COUNT, TOKEN_COLORS, STANDARD_COLORS, TOTAL_ACTIONS

def simplified_legal_take3(bank):
    """
    Simplified Take 3 legal move logic:
    - 3+ available: Take 3 legal if all 3 colors are available
    - 2 available: Take 3 legal if the available colors are subset of the 3 colors  
    - 1 available: Take 3 legal if the available color is in the 3 colors
    """
    mask = [0] * TAKE3_COUNT
    available_colors = set([i for i in range(5) if bank[i] >= 1])
    
    for idx, (a, b, c) in enumerate(TAKE3_COMBOS):
        action_colors = set([a, b, c])
        
        if len(available_colors) >= 3:
            # All 3 colors must be available
            if action_colors.issubset(available_colors):
                mask[idx] = 1
        elif len(available_colors) >= 1:
            # Available colors must be subset of action colors
            if available_colors.issubset(action_colors):
                mask[idx] = 1
                
    return mask

def current_legal_take3(bank):
    """Current complex logic from the engine"""
    from splendor_gym.engine.rules import _partial_take3_index_for_available
    
    mask = [0] * TAKE3_COUNT
    available_colors = [i for i in range(5) if bank[i] >= 1]
    
    if len(available_colors) >= 3:
        for idx, (a, b, c) in enumerate(TAKE3_COMBOS):
            if bank[a] >= 1 and bank[b] >= 1 and bank[c] >= 1:
                mask[idx] = 1
    elif len(available_colors) > 0:
        mapped = _partial_take3_index_for_available(available_colors)
        if mapped is not None:
            mask[mapped - TAKE3_OFFSET] = 1
            
    return mask

def simulate_take3_execution(bank, action_idx):
    """Simulate how the action would execute under simplified logic"""
    available_colors = [i for i in range(5) if bank[i] >= 1]
    a, b, c = TAKE3_COMBOS[action_idx]
    action_colors = [a, b, c]
    
    # Take only the colors that are both in action and available
    tokens_taken = []
    new_bank = bank.copy()
    
    for color in action_colors:
        if color in available_colors and new_bank[color] > 0:
            new_bank[color] -= 1
            tokens_taken.append(color)
            
    return tokens_taken, new_bank

# Test scenarios
test_scenarios = [
    ([4, 4, 4, 4, 4, 5], "Full bank (5 colors)"),
    ([2, 1, 0, 0, 0, 5], "2 colors: white, blue"),  
    ([0, 0, 0, 3, 0, 5], "1 color: red"),
    ([1, 1, 1, 0, 0, 5], "3 colors: white, blue, green"),
    ([0, 0, 2, 2, 0, 5], "2 colors: green, red"),
]

print("=== COMPARISON: Current vs Simplified Take 3 Logic ===\n")

for bank, description in test_scenarios:
    print(f"SCENARIO: {description}")
    print(f"Bank: {[f'{STANDARD_COLORS[i]}:{bank[i]}' for i in range(5)]}")
    
    current_mask = current_legal_take3(bank)
    simplified_mask = simplified_legal_take3(bank)
    
    current_legal = [i for i, val in enumerate(current_mask) if val == 1]
    simplified_legal = [i for i, val in enumerate(simplified_mask) if val == 1]
    
    print(f"Current legal actions ({len(current_legal)}): {current_legal}")
    print(f"Simplified legal actions ({len(simplified_legal)}): {simplified_legal}")
    
    # Show what each simplified legal action would do
    if simplified_legal:
        print("Simplified action results:")
        for action_idx in simplified_legal[:5]:  # Show first 5 to avoid clutter
            combo = TAKE3_COMBOS[action_idx]
            combo_colors = [STANDARD_COLORS[i] for i in combo]
            tokens_taken, new_bank = simulate_take3_execution(bank, action_idx)
            taken_colors = [STANDARD_COLORS[i] for i in tokens_taken]
            print(f"  Action {action_idx} {combo_colors} → takes {taken_colors}")
    
    print("-" * 60)
