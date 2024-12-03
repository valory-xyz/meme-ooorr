// SPDX-License-Identifier: MIT
pragma solidity ^0.8.28;

import {PriceOracle} from "./PriceOracle.sol";

interface IUniswapV2 {
    function token0() external view returns (address);
    function getReserves() external view returns (uint112 _reserve0, uint112 _reserve1, uint32 _blockTimestampLast);
    function price0CumulativeLast() external view returns (uint256);
    // function price1CumulativeLast() external view returns (uint256);
}

/// @title UniPriceOracle - a smart contract oracle wrapper for Uni V2 pools
/// @dev This contract acts as an oracle wrapper for a specific Uni V2 pool. It allows:
///      1) Getting the price by any caller
///      2) Validating slippage against the oracle
contract UniPriceOracle {
    // LP token address
    address public immutable pair;
    // Max allowable slippage
    uint256 public immutable maxSlippage;
    // LP token direction
    uint256 public immutable direction;

    constructor(
        address _nativeToken,
        uint256 _maxSlippage,
        address _pair
    ) {
        pair = _pair;
        maxSlippage = _maxSlippage;

        // Get token direction
        address token0 =  IUniswapV2(pair).token0();
        if (token0 != _nativeToken) {
            direction = 1;
        }
    }

    /// @dev Gets the current OLAS token price in 1e18 format.
    function getPrice() public view returns (uint256) {
        uint256[] memory balances = new uint256[](2);
        (balances[0], balances[1], ) = IUniswapV2(pair).getReserves();
        // Native token
        uint256 balanceIn = balances[direction];
        // OLAS
        uint256 balanceOut = balances[(direction + 1) % 2];

        return (balanceOut * 1e18) / balanceIn;
    }

    /// @dev Updates the time-weighted average price.
    function updatePrice() external view returns (bool) {
        // Nothing to update; use built-in TWAP from uniswap v2 pool
        return true;
    }

    /// @dev Validates the current price against a TWAP according to slippage tolerance.
    /// @param slippage the acceptable slippage tolerance
    function validatePrice(uint256 slippage) external view returns (bool) {
        require(slippage <= 100, "Slippage must be <= 100%");

        // Compute time-weighted average price
        // Fetch the cumulative prices from the pair
        uint256 price0Cumulative = IUniswapV2(pair).price0CumulativeLast();
        // uint256 price1Cumulative = IUniswapV2(pairAddress).price1CumulativeLast();

        // Fetch the reserves and the last block timestamp
        (, , uint32 blockTimestampLast) = IUniswapV2(pair).getReserves();

        // Fetch the current block timestamp
        uint32 blockTimestamp = uint32(block.timestamp);

        // Require at least one block since last update
        if (blockTimestamp > blockTimestampLast) return false;
        uint256 actualTimeElapsed = blockTimestamp - blockTimestampLast;

        // TODO: get correct TWAP based on direction
        // Calculate the TWAP for token0 in terms of token1
        uint256 timeWeightedAverage = (price0Cumulative / actualTimeElapsed);
        // TODO // Calculate the TWAP for token1 in terms of token0
        // TODO timeWeightedAverage = (price1Cumulative / actualTimeElapsed);

        uint256 tradePrice = getPrice();

        // Validate against slippage thresholds
        uint256 lowerBound = (timeWeightedAverage * (100 - slippage)) / 100;
        uint256 upperBound = (timeWeightedAverage * (100 + slippage)) / 100;

        return tradePrice >= lowerBound && tradePrice <= upperBound;
    }
}
