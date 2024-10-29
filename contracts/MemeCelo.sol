// SPDX-License-Identifier: MIT
pragma solidity ^0.8.28;

import {MemeFactory} from "./MemeFactory.sol";

// Bridge interface
interface IBridge {
    /// @dev Transfers tokens through Wormhole portal.
    function transferTokens(
        address token,
        uint256 amount,
        uint16 recipientChain,
        bytes32 recipient,
        uint256 arbiterFee,
        uint32 nonce
    ) external payable returns (uint64 sequence);
}

// ERC20 interface
interface IERC20 {
    /// @dev Sets `amount` as the allowance of `spender` over the caller's tokens.
    /// @param spender Account address that will be able to transfer tokens on behalf of the caller.
    /// @param amount Token amount.
    /// @return True if the function execution is successful.
    function approve(address spender, uint256 amount) external returns (bool);
}

// Oracle interface
interface IOracle {
    /// @dev Gets latest round token price data.
    function latestRoundData()
        external returns (uint80 roundId, int256 answer, uint256 startedAt, uint256 updatedAt, uint80 answeredInRound);
}

// UniswapV2 interface
interface IUniswap {
    /// @dev Swaps exact amount of ETH for a specified token.
    function swapExactETHForTokens(uint256 amountOutMin, address[] calldata path, address to, uint256 deadline)
        external payable returns (uint256[] memory amounts);
}

/// @title MemeCelo - a smart contract factory for Meme Token creation on Celo.
abstract contract MemeCelo is MemeFactory {
    // Slippage parameter (3%)
    uint256 public constant SLIPPAGE = 97;
    // Ethereum mainnet chain Id in Wormhole format
    uint16 public constant WORMHOLE_ETH_CHAIN_ID = 2;

    // L2 token relayer bridge address
    address public immutable l2TokenRelayer;
    // Oracle address
    address public immutable oracle;

    // Contract nonce
    uint256 public nonce;

    /// @dev MemeBase constructor
    constructor(
        address _olas,
        address _usdc,
        address _weth,
        address _router,
        address _factory,
        address _balancerVault,
        uint256 _minNativeTokenValue,
        address _l2TokenRelayer,
        address _oracle
    ) MemeFactory(_olas, _usdc, _weth, _router, _factory, _balancerVault, _minNativeTokenValue) {
        l2TokenRelayer = _l2TokenRelayer;
        oracle = _oracle;
    }

    /// @dev Buys USDC on UniswapV2 using Celo amount.
    /// @param celoAmount Input Celo amount.
    /// @return Stable token amount bought.
    function _buyStableToken(uint256 celoAmount, uint256) internal override returns (uint256) {
        address[] memory path = new address[](2);
        path[0] = weth;
        path[1] = usdc;

        // Calculate price by Oracle
        (, int256 answerPrice, , , ) = IOracle(oracle).latestRoundData();
        require(answerPrice > 0, "Oracle price is incorrect");

        // Oracle returns 8 decimals, USDC has 6 decimals, need to additionally divide by 100
        // ETH: 18 decimals, USDC leftovers: 2 decimals, percentage: 2 decimals; denominator = 18 + 2 + 2 = 22
        uint256 limit = uint256(answerPrice) * celoAmount * SLIPPAGE / 1e22;
        // Swap ETH for USDC
        uint256[] memory amounts = IUniswap(router).swapExactETHForTokens{ value: celoAmount }(
            limit,
            path,
            address(this),
            block.timestamp
        );

        // Return the USDC amount bought
        return amounts[1];
    }

    /// @dev Buys OLAS on UniswapV2.
    /// @param usdcAmount USDC amount.
    /// @return Obtained OLAS amount.
    function _buyOLAS(uint256 usdcAmount, uint256 limit) internal override returns (uint256) {
        address[] memory path = new address[](2);
        path[0] = usdc;
        path[1] = olas;

        // Calculate price by Oracle
        //(, int256 answerPrice, , , ) = IOracle(oracle).latestRoundData();
        //require(answerPrice > 0, "Oracle price is incorrect");

        // Swap USDC for
        uint256[] memory amounts = IUniswap(router).swapExactETHForTokens{ value: usdcAmount }(
            limit,
            path,
            address(this),
            block.timestamp
        );

        // Return the USDC amount bought
        return amounts[1];
    }

    /// @dev Bridges OLAS amount back to L1 and burns.
    /// @param OLASAmount OLAS amount.
    /// @return msg.value leftovers if partially utilized by the bridge.
    function _bridgeAndBurn(uint256 OLASAmount, uint256, bytes memory) internal override returns (uint256) {
        // Approve bridge to use OLAS
        IERC20(olas).approve(l2TokenRelayer, OLASAmount);

        // Bridge arguments
        bytes32 olasBurner = bytes32(uint256(uint160(OLAS_BURNER)));
        uint256 localNonce = nonce;

        // Bridge OLAS to mainnet to get burned
        IBridge(l2TokenRelayer).transferTokens(olas, OLASAmount, WORMHOLE_ETH_CHAIN_ID, olasBurner, 0, uint32(nonce));

        // Adjust nonce
        nonce = localNonce + 1;

        emit OLASJourneyToAscendance(olas, OLASAmount);

        return msg.value;
    }
}