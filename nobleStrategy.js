

import { AIStrategyInterface } from '../baseStrategy.js';
import { ActionGenerator } from '../../engine/gameFlow.js';
import { gameStateDebugger } from '../gameStateDebugger.js';

export class NobleStrategy extends AIStrategyInterface {
  constructor(config = {}) {
    super(config);
    this.name = 'NobleStrategy';
    this.version = '2.0.0';
    this.description = 'Noble strategy with structured card and token evaluation';
    this.debug = config.debug || false;
    
    // Debug the configuration being passed
    if (this.debug) {
      console.log('🔧 NobleStrategy constructor called with config:', config);
    }
  }

  async makeDecision(gameState, context = {}) {
    const playerIndex = context.playerIndex || gameState.currentPlayer;
    
    // Check for debug configuration in context as well
    const shouldDebug = this.debug || (context.strategyConfig && context.strategyConfig.debug);
    
    const validActionsObj = ActionGenerator.getValidActions(gameState);
    const allActions = this.flattenValidActions(validActionsObj, gameState);
    
    // Validate that all cards have proper IDs
    this.validateCardIds(gameState, allActions);
    
    // Get player's token values and card evaluations
    const { tokenValues: playerTokenValues, cardEvaluations: playerCardEvaluations } = 
      this.calculateTokenValuesAndCardEvaluations(gameState, playerIndex, shouldDebug);
    
    // Check for winning moves first (including noble points)
    const purchaseActions = allActions.filter(action => action.type === 'purchase_card');
    const winningMove = this.findWinningMove(gameState, playerIndex, purchaseActions, shouldDebug);
    
    if (winningMove) {
      // Debug winning move
      if (shouldDebug) {
        console.log('🔍 NobleStrategy Debug:');
        console.log('Game State:', gameStateDebugger.displayGameState(gameState));
        console.log('🎯 WINNING MOVE FOUND!');
        console.log('Winning Action:', gameStateDebugger.displayAction(winningMove.action));
        console.log('Points Achieved:', winningMove.points);
        console.log('---');
      }
      
      return {
        action: winningMove.action,
        confidence: 'high',
        reasoning: `Winning move:`
      };
    }

    // --- NEW LOGIC: Prioritize reserve or token-taking for almost-winning cards ---
    const almostWinningMove = this.handleAlmostWinningMove(gameState, playerIndex, allActions, shouldDebug);
    if (almostWinningMove) {
      return almostWinningMove;
    }
    // --- END NEW LOGIC ---
    
    // Get opponent's card evaluations for defensive considerations
    const opponentIndex = 1 - playerIndex;
    const { tokenValues: opponentTokenValues, cardEvaluations: opponentCardEvaluations } = 
      this.calculateTokenValuesAndCardEvaluations(gameState, opponentIndex, false);
    
    // Debug information
    if (shouldDebug) {
      console.log('🔍 NobleStrategy Debug:');
      console.log('Game State:', gameStateDebugger.displayGameState(gameState));
      console.log('Token Values:', gameStateDebugger.displayTokenValues(playerTokenValues));
    }
    
    // Evaluate all actions
    const evaluatedActions = allActions.map(action => {
      let value = 0;
      let actionType = 'unknown';

      if (action.type === 'take_and_discard_tokens') {
        value = this.evaluateTakeAction(action, playerTokenValues);
        actionType = 'value_based_take';
      } else if (action.type === 'purchase_card') {
        const cardId = action.card.id;
        const cardEvaluation = playerCardEvaluations[cardId];
        value = cardEvaluation.value;
        
        // Add defensive bonus if this is the opponent's top card
        const defensiveValue = this.getDefensiveValue(action.card, opponentCardEvaluations, 'purchase');
        value += defensiveValue;
        
        // Debug: Show card value breakdown for purchase actions
        if (shouldDebug) {
          console.log(`  Pre-calculated Total Value: ${value.toFixed(2)}`);
          console.log(`  Card Details: ${action.card.color} color, ${action.card.points} points, tier ${action.card.tier}`);
          console.log(`  Cost: ${JSON.stringify(action.card.cost)}`);
        }
        
        actionType = 'structured_purchase';
      } else if (action.type === 'reserve_and_discard') {
        value = this.evaluateReserveAction(action, playerCardEvaluations, opponentCardEvaluations, gameState, playerIndex, playerTokenValues);
        
        actionType = 'structured_reserve';
      }

      return { action, value, actionType };
    });
    
    // Sort by value and select the best action
    evaluatedActions.sort((a, b) => b.value - a.value);
    const selectedAction = evaluatedActions[0];
    
    // Debug selected action
    if (shouldDebug) {
      console.log('Selected Action:', gameStateDebugger.displayAction(selectedAction.action));
      console.log('Action Value:', selectedAction.value.toFixed(2));
      console.log('Action Type:', selectedAction.actionType);
      console.log('---');
    }
    
    return {
      action: selectedAction.action,
      confidence: 'medium',
      reasoning: `Noble strategy selection: ${selectedAction.actionType} - ${selectedAction.action.type} (value: ${selectedAction.value.toFixed(2)})`
    };
  }


  calculateTokenValuesAndCardEvaluations(gameState, playerIndex, shouldDebug) {
    let tokenValues = { white: 2, blue: 2, green: 2, red: 2, black: 2, gold: 2 };
    let cardEvaluations = this.evaluateAllCards(gameState, playerIndex, tokenValues, shouldDebug);
    tokenValues = this.evaluateTokens(cardEvaluations, gameState, playerIndex);
    cardEvaluations = this.evaluateAllCards(gameState, playerIndex, tokenValues, shouldDebug);
    tokenValues = this.evaluateTokens(cardEvaluations, gameState, playerIndex);
    cardEvaluations = this.evaluateAllCards(gameState, playerIndex, tokenValues, shouldDebug);
    tokenValues = this.evaluateTokens(cardEvaluations, gameState, playerIndex);
    
    // Debug: Show final card evaluations with value breakdown
    if (shouldDebug) {
      console.log('🔍 NobleStrategy Final Card Evaluations:');
      Object.values(cardEvaluations).forEach(cardEval => {
        const card = cardEval.card;
        const costStr = this.formatCost(card.cost);
        console.log(`  ${card.id} (${card.color}, ${card.points}p, tier ${card.tier}) [${costStr}]: Total Value = ${cardEval.value.toFixed(2)}`);
      });
      console.log('---');
    }
    
    return { tokenValues, cardEvaluations };
  }



  evaluateAllCards(gameState, playerIndex, tokenValues, shouldDebug) {
    const allCards = this.getAllAvailableCards(gameState, playerIndex);
    const cardEvaluations = {};
    
    allCards.forEach(card => {
      const cardId = card.id;
      
      const totalValue = this.evaluateCard(gameState, card, playerIndex, tokenValues, allCards, shouldDebug);
      
      cardEvaluations[cardId] = {
        card,
        value: totalValue,
        color: card.color,
        points: card.points,
        tier: card.tier
      };
      
    });
    
    return cardEvaluations;
  }

  evaluateCard(gameState, card, playerIndex, tokenValues, allAvailableCards = [], shouldDebug = false) {
    const nobleValue = this.calculateNobleValue(gameState, card, playerIndex);
    const discountValue = this.calculateDiscountValue(card, tokenValues, gameState, playerIndex);
    const tokenCostValue = this.calculateTokenCostValue(gameState, card, playerIndex);
    const currentPlayerCardCount = this.getCurrentPlayerCardCount(gameState, gameState.currentPlayer);
    const pointsValue = this.calculatePointsValue(card, currentPlayerCardCount);
    
    const totalValue = nobleValue + discountValue + tokenCostValue + pointsValue;
    
    return totalValue;
  }

  calculateNobleValue(gameState, card, playerIndex) {
    const cardColor = card.color;
    let nobleValue = 0;
    
    gameState.board.nobles.forEach(noble => {
      if (noble.requirement && noble.requirement[cardColor] > 0) {
        const player = gameState.players[playerIndex];
        
        // Check if this specific color requirement is not satisfied yet
        const required = noble.requirement[cardColor];
        const owned = player.cards[cardColor];
        const colorNeeded = Math.max(0, required - owned);
        
        // Only add noble value if this color is still needed
        if (colorNeeded > 0) {
          nobleValue += 1;
          
          let cardsNeeded = 0;
          
          Object.keys(noble.requirement).forEach(color => {
            const required = noble.requirement[color];
            const owned = player.cards[color];
            const needed = Math.max(0, required - owned);
            cardsNeeded += needed;
          });
          
          // Add extra 1 if cardsNeeded is between 4-6
          if (cardsNeeded >= 4 && cardsNeeded <= 6) {
            nobleValue += 1;
          }
          // Add bonus for cardsNeeded of 1, 2, or 3
          else if (cardsNeeded === 3) {
            nobleValue += 3; // 3 away: +3 bonus
          } else if (cardsNeeded === 2) {
            nobleValue += 7; // 2 away: +7 bonus
          } else if (cardsNeeded === 1) {
            nobleValue += 12; // 1 away: +12 bonus
          }
        }
      }
    });
    
    return nobleValue;
  }

  calculateDiscountValue(card, tokenValues, gameState, playerIndex) {
    const cardColor = card.color;
    
    const currentPlayerCardCount = this.getCurrentPlayerCardCount(gameState, gameState.currentPlayer);
    let discountValue = tokenValues[cardColor];
    discountValue += 15 - 0.5 * currentPlayerCardCount;
    
    const player = gameState.players[playerIndex];
    if (player && player.cards && player.cards[cardColor]) {
      const sameColorCards = player.cards[cardColor];
      discountValue -= 2 * sameColorCards;
    }

    // Ensure discount value is never negative
    return Math.max(0, discountValue);
  }

 calculateTokenCostValue(gameState, card, playerIndex) {
    const player = gameState.players[playerIndex];
    const playerCards = player.cards;
    const playerTokens = player.tokens;
    const tokenBank = gameState.tokens;
    const cardCost = card.cost;
    
    // If card has no cost (like owned cards), return 0
    if (!cardCost) {
      return 0;
    }
    
    let totalCostValue = 0;
    
    Object.keys(cardCost).forEach(color => {
      const cost = cardCost[color];
      const discount = playerCards[color];
      const tokens = playerTokens[color];
      const tokensNeeded = Math.max(0, cost - discount);
      const tokensNeededWithBank = Math.max(0, cost - discount - tokens);
      const availableInBank = tokenBank[color];
      
      const costValue = -1.5 * tokensNeeded - 0.5 * tokensNeeded ** 1.5;
      totalCostValue += costValue;
    });
    
    return totalCostValue;
  }

  calculatePointsValue(card, currentPlayerCardCount) {
    return card.points * (1 + 0.1 * currentPlayerCardCount);
  }

  evaluateTokens(cardEvaluations, gameState, playerIndex) {
    const tokenValues = { white: 1, blue: 1, green: 1, red: 1, black: 1, gold: 1 };
    
    // Get all cards that have require1 > 0
    const allCards = Object.values(cardEvaluations);
    const cardsWithRequire1 = allCards.filter(cardEvaluation => {
      const card = cardEvaluation.card;
      if (!card.cost) return false;
      
      return Object.entries(card.cost).some(([color, amount]) => {
        if (color === 'gold' || amount <= 0) return false;
        const demand = amount;
        const discount = this.getPlayerDiscount(color, gameState, playerIndex);
        const require1 = Math.max(0, demand - discount);
        return require1 > 0;
      });
    });
    
    // Sort cards by value for ranking
    const rankedCards = cardsWithRequire1.sort((a, b) => b.value - a.value);
    
    // Apply bonuses to tokens based on relative rank
    rankedCards.forEach((cardEvaluation, index) => {
      const card = cardEvaluation.card;
      if (card.cost) {
        Object.entries(card.cost).forEach(([color, amount]) => {
          if (color !== 'gold' && amount > 0) {
            const demand = amount;
            const discount = this.getPlayerDiscount(color, gameState, playerIndex);
            const inhand = this.getPlayerTokens(color, gameState, playerIndex);
            
            const require1 = Math.max(0, demand - discount);
            const require2 = Math.max(0, demand - discount - inhand);
            
            if (require1 > 0) {
              const baseBonus = this.calculateRankBonus(index, rankedCards.length);
              this.applyTokenBonus(tokenValues, color, baseBonus, require2 > 0);
            }
          }
        });
      }
    });
    
    return tokenValues;
  }

  calculateRankBonus(rank, totalCards) {
    // Calculate bonus based on relative rank (0 = top card)
    if (rank === 0) return 1.0;
    if (rank === 1) return 0.5;
    if (rank === 2) return 0.3;
    if (rank === 3) return 0.2;
    // For ranks beyond 4, use a diminishing formula
    return Math.max(0.05, 0.16 * Math.pow(0.8, rank - 3));
  }

  applyTokenBonus(tokenValues, color, baseBonus, isRequire2) {
    if (isRequire2) {
      tokenValues[color] += baseBonus; // Full bonus for require2
    } else {
      tokenValues[color] += baseBonus * 0.5; // Half bonus for require1 only
    }
  }

  getPlayerDiscount(color, gameState, playerIndex) {
    const player = gameState.players[playerIndex];
    return player.cards[color];
  }

  getPlayerTokens(color, gameState, playerIndex) {
    const player = gameState.players[playerIndex];
    return player.tokens[color];
  }

  evaluateTakeAction(action, tokenValues) {
    let value = 0;
    
    if (action.tokens) {
      Object.keys(action.tokens).forEach(color => {
        const count = action.tokens[color];
        const tokenValue = tokenValues[color];
        value += count * tokenValue;
      });
    }

    if (action.discard) {
      Object.keys(action.discard).forEach(color => {
        const count = action.discard[color];
        const tokenValue = tokenValues[color];
        value -= count * tokenValue;
      });
    }
    
    return value;
  }

  evaluateReserveAction(action, playerCardEvaluations, opponentCardEvaluations, gameState, playerIndex, tokenValues) {
    if (!action.card) {
      return -100;
    }

    const playerSortedCards = Object.values(playerCardEvaluations)
      .sort((a, b) => b.value - a.value);
    const opponentSortedCards = Object.values(opponentCardEvaluations)
      .sort((a, b) => b.value - a.value);

    const playerTopCard = playerSortedCards[0];
    const opponentTopCard = opponentSortedCards[0];
    
    const isPlayerTop = action.card.id === playerTopCard.card.id;
    const isOpponentTop = action.card.id === opponentTopCard.card.id;

    if (!isPlayerTop && !isOpponentTop) {
      return -100;
    }

    let value = 0;
    value += 3; // Base gold token value

    // Calculate value based on which top card it is
    if (isPlayerTop && isOpponentTop) {
      // Both players' top card - take the higher value
      const playerValue = playerSortedCards.length >= 2 ? 
        (playerTopCard.value - playerSortedCards[1].value) * 0.5 : playerTopCard.value * 0.5;
      const opponentValue = opponentSortedCards.length >= 2 ? 
        (opponentTopCard.value - opponentSortedCards[1].value) * 0.5 : opponentTopCard.value * 0.5;
      value += Math.max(playerValue, opponentValue);
    } else if (isPlayerTop) {
      // Only player's top card
      if (playerSortedCards.length >= 2) {
        const difference = playerTopCard.value - playerSortedCards[1].value;
        value += difference * 0.5;
      }
    } else if (isOpponentTop) {
      // Only opponent's top card
      if (opponentSortedCards.length >= 2) {
        const difference = opponentTopCard.value - opponentSortedCards[1].value;
        value += difference * 0.5;
      }
    }

    if (action.discard) {
      Object.keys(action.discard).forEach(color => {
        const count = action.discard[color];
        const tokenValue = tokenValues[color];
        value -= count * tokenValue;
      });
    }

    return value;
  }

    getDefensiveValue(card, opponentCardEvaluations, actionType) {
      if (!card || !opponentCardEvaluations || actionType !== 'purchase') {
        return 0;
      }
      
      const opponentCardList = Object.values(opponentCardEvaluations);
      if (opponentCardList.length === 0) {
        return 0;
      }
      
      opponentCardList.sort((a, b) => b.value - a.value);
      const opponentTopCard = opponentCardList[0];
      const opponentSecondCard = opponentCardList.length > 1 ? opponentCardList[1] : null;
      
      if (card.id !== opponentTopCard.card.id) {
        return 0;
      }
      
      const topSecondDifference = opponentSecondCard ? 
        opponentTopCard.value - opponentSecondCard.value : opponentTopCard.value;
      
      return 0.5 * topSecondDifference;
  }


  flattenValidActions(validActions, gameState) {
    const actions = [];
    actions.push(...validActions.takeAndDiscardTokens);
    actions.push(...validActions.purchaseCard);
    actions.push(...validActions.reserveAndDiscardCard);
    return actions;
  }

  validateCardIds(gameState, allActions) {
    const missingIds = [];
    
    // Check cards in game state
    const allCards = this.getAllAvailableCards(gameState);
    allCards.forEach(card => {
      if (!card.id) {
        missingIds.push(`Card in game state: ${JSON.stringify(card)}`);
      }
    });
    
    // Check cards in actions
    allActions.forEach(action => {
      if (action.type === 'purchase_card' && action.card && !action.card.id) {
        missingIds.push(`Card in purchase action: ${JSON.stringify(action.card)}`);
      }
    });
    
    if (missingIds.length > 0) {
      throw new Error(`Cards missing IDs in NobleStrategy: ${missingIds.join(', ')}`);
    }
  }

  getAllAvailableCards(gameState, playerIndex = null) {
    const cards = [];
    
    // Add cards from all tiers on the board (visible cards)
    if (gameState.board.tier1) cards.push(...gameState.board.tier1);
    if (gameState.board.tier2) cards.push(...gameState.board.tier2);
    if (gameState.board.tier3) cards.push(...gameState.board.tier3);
    
    // Add reserved cards from specified player (or current player if not specified)
    const targetPlayerIndex = playerIndex !== null ? playerIndex : gameState.currentPlayer;
    if (gameState.players && gameState.players[targetPlayerIndex]) {
      const targetPlayer = gameState.players[targetPlayerIndex];
      if (targetPlayer.reserved) {
        cards.push(...targetPlayer.reserved);
      }
    }
    
    return cards;
  }

  getCurrentPlayerCardCount(gameState, playerIndex) {
    const player = gameState.players[playerIndex];
    if (!player || !player.cards) return 0;
    
    let totalCards = 0;
    Object.values(player.cards).forEach(count => {
      totalCards += count;
    });
    
    return 2 * totalCards;
  }

  formatCost(cost) {
    if (!cost) return '';
    const parts = [];
    const colorMap = { white: 'w', blue: 'b', green: 'g', red: 'r', black: 'k' };
    Object.entries(cost).forEach(([color, amount]) => {
      if (amount > 0) {
        parts.push(`${amount}${colorMap[color]}`);
      }
    });
    return parts.join('');
  }

  handleAlmostWinningMove(gameState, playerIndex, allActions, shouldDebug) {
    const targetPoints = 15;
    const player = gameState.players[playerIndex];
    const allAvailableCards = this.getAllAvailableCards(gameState);
    // 1. Reserve: Any card that would win (including nobles) and is missing exactly 1 token
    const reserveCandidates = allAvailableCards.filter(card => {
      if (!card.cost || !card.points) return false;
      const pointsAfterCard = player.points + card.points;
      const noblesGained = this.simulateNoblesAfterPurchase(gameState, playerIndex, card);
      // Only count at most 1 noble
      const totalPoints = pointsAfterCard + (noblesGained.length > 0 ? 3 : 0);
      if (totalPoints < targetPoints) return false;
      let missing = 0;
      for (const color of Object.keys(card.cost)) {
        const cost = card.cost[color];
        const discount = player.cards[color];
        const tokens = player.tokens[color];
        const need = Math.max(0, cost - discount - tokens);
        missing += need;
      }
      return missing === 1;
    });
    if (reserveCandidates.length > 0) {
      const reserveActions = allActions.filter(a => a.type === 'reserve_and_discard');
      for (const card of reserveCandidates) {
        const reserve = reserveActions.find(a => a.card && a.card.id === card.id);
        if (reserve) {
          if (shouldDebug) {
            console.log('🔍 NobleStrategy Debug: Reserving almost-winning card:', card);
          }
          return {
            action: reserve,
            confidence: 'high',
            reasoning: `Reserving card that is one token short of a winning purchase (including possible noble).`
          };
        }
      }
    }
    // 2. Token-taking: For cards missing more than 1, but all missing colors are missing by exactly 1, and all those tokens are available in the bank
    const takeActions = allActions.filter(a => a.type === 'take_and_discard_tokens');
    for (const card of allAvailableCards) {
      if (!card.cost || !card.points) continue;
      const pointsAfterCard = player.points + card.points;
      const noblesGained = this.simulateNoblesAfterPurchase(gameState, playerIndex, card);
      // Only count at most 1 noble
      const totalPoints = pointsAfterCard + (noblesGained.length > 0 ? 3 : 0);
      if (totalPoints < targetPoints) continue;
      // Find all colors where we are missing exactly 1
      const missingColors = [];
      let allMissingAreOne = true;
      for (const color of Object.keys(card.cost)) {
        const cost = card.cost[color];
        const discount = player.cards[color];
        const tokens = player.tokens[color];
        const need = Math.max(0, cost - discount - tokens);
        if (need === 1) {
          // Only consider if token is available in the bank
          if (gameState.tokens[color] < 1) {
            allMissingAreOne = false;
            break;
          }
          missingColors.push(color);
        } else if (need > 1) {
          allMissingAreOne = false;
          break;
        }
      }
      if (missingColors.length > 1 && allMissingAreOne && missingColors.length <= 3) {
        // Is there a take action that gives us at least one of these colors?
        const take = takeActions.find(a => missingColors.some(color => a.tokens && a.tokens[color] > 0));
        if (take) {
          if (shouldDebug) {
            console.log('🔍 NobleStrategy Debug: Taking token(s) towards almost-winning card:', card, 'Missing colors:', missingColors);
          }
          return {
            action: take,
            confidence: 'high',
            reasoning: `Taking token(s) (${missingColors.join(', ')}) to enable winning purchase next turn (including possible noble).`
          };
        }
      }
    }
    // If nothing found
    return null;
  }

  // Helper to simulate which nobles would be gained if the player bought a given card
  simulateNoblesAfterPurchase(gameState, playerIndex, card) {
    const player = gameState.players[playerIndex];
    // Simulate new card counts after purchase
    const newCardCounts = { ...player.cards };
    if (!newCardCounts[card.color]) newCardCounts[card.color] = 0;
    newCardCounts[card.color] += 1;
    // Check which nobles would be gained
    const nobles = gameState.board.nobles || [];
    const gained = [];
    for (const noble of nobles) {
      let qualifies = true;
      for (const color of Object.keys(noble.requirement)) {
        const required = noble.requirement[color];
        const owned = newCardCounts[color] || 0;
        if (owned < required) {
          qualifies = false;
          break;
        }
      }
      if (qualifies) gained.push(noble);
    }
    return gained;
  }

  // Find winning moves including noble points
  findWinningMove(gameState, playerIndex, purchaseActions, shouldDebug = false) {
    const targetPoints = 15;
    const player = gameState.players[playerIndex];
    const winningMoves = [];
    
    // Check each purchase action to see if it would result in a win
    for (const action of purchaseActions) {
      const pointsAfterCard = player.points + action.card.points;
      const noblesGained = this.simulateNoblesAfterPurchase(gameState, playerIndex, action.card);
      const totalPoints = pointsAfterCard + (noblesGained.length > 0 ? 3 : 0);
      
      if (shouldDebug) {
        console.log(`🔍 Checking card ${action.card.id}: ${action.card.points} points + ${noblesGained.length > 0 ? 3 : 0} noble = ${totalPoints} total`);
      }
      
      if (totalPoints >= targetPoints) {
        winningMoves.push({
          action: action,
          points: totalPoints,
          cardPoints: action.card.points,
          noblePoints: noblesGained.length > 0 ? 3 : 0
        });
      }
    }
    
    // If no winning moves found
    if (winningMoves.length === 0) {
      return null;
    }
    
    // Sort winning moves by total points (highest first)
    winningMoves.sort((a, b) => b.points - a.points);
    
    if (shouldDebug && winningMoves.length > 1) {
      console.log(`🎯 Found ${winningMoves.length} winning moves:`);
      winningMoves.forEach((move, index) => {
        console.log(`  ${index + 1}. ${move.action.card.id}: ${move.cardPoints} + ${move.noblePoints} = ${move.points} points`);
      });
      console.log(`🏆 Selected highest scoring move: ${winningMoves[0].action.card.id} with ${winningMoves[0].points} points`);
    }
    
    // Return the winning move with the highest points
    return winningMoves[0];
  }
}