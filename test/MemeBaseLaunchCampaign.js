/*global describe, context, beforeEach, it*/
const { expect } = require("chai");
const { ethers } = require("hardhat");
const helpers = require("@nomicfoundation/hardhat-network-helpers");

// This works on a fork only!
const main = async () => {
    const fs = require("fs");
    const globalsFile = "globals.json";
    let dataFromJSON = fs.readFileSync(globalsFile, "utf8");
    const parsedData = JSON.parse(dataFromJSON);

    const name = "Meme";
    const symbol = "MM";
    const totalSupply = "1" + "0".repeat(24);
    const smallDeposit = ethers.utils.parseEther("1");
    const defaultDeposit = ethers.utils.parseEther("1500");
    const oneDay = 86400;
    const twoDays = 2 * oneDay;
    // Nonce 1 is reserved for the campaign token
    // Nonce 2 is the first new meme token
    const nonce0 = 1;
    const nonce1 = 2;
    const nonce2 = 3;

    signers = await ethers.getSigners();
    deployer = signers[0];

    console.log("Getting launch campaign data");
    const campaignFile = "scripts/deployment/memebase_campaign.json";
    dataFromJSON = fs.readFileSync(campaignFile, "utf8");
    const campaignData = JSON.parse(dataFromJSON);
    console.log("Number of entries:", campaignData.length);
    
    const accounts = new Array();
    const amounts = new Array();
    for (let i = 0; i < campaignData.length; i++) {
        accounts.push(campaignData[i]["hearter"]);
        amounts.push(campaignData[i]["amount"].toString());
    }

    // BuyBackBurner implementation and proxy
    const BuyBackBurner = await ethers.getContractFactory("BuyBackBurner");
    const buyBackBurnerImplementation = await BuyBackBurner.deploy();
    await buyBackBurnerImplementation.deployed();

    // Initialize buyBackBurner
    const proxyData = buyBackBurnerImplementation.interface.encodeFunctionData("initialize", []);
    const BuyBackBurnerProxy = await ethers.getContractFactory("BuyBackBurnerProxy");
    const buyBackBurnerProxy = await BuyBackBurnerProxy.deploy(buyBackBurnerImplementation.address, proxyData);
    await buyBackBurnerProxy.deployed();

    const buyBackBurner = await ethers.getContractAt("BuyBackBurner", buyBackBurnerProxy.address);

    const MemeBase = await ethers.getContractFactory("MemeBase");
    const memeBase = await MemeBase.deploy(parsedData.olasAddress, parsedData.wethAddress,
        parsedData.uniV3positionManagerAddress, buyBackBurner.address, parsedData.minNativeTokenValue,
        accounts, amounts);
    await memeBase.deployed();

    // Summon a new meme token - negative cases
    const minNativeTokenValue = await memeBase.minNativeTokenValue();
    const MIN_TOTAL_SUPPLY = await memeBase.MIN_TOTAL_SUPPLY();
    const uint128MaxPlusOne = BigInt(2) ** BigInt(128);
    await expect(
        memeBase.summonThisMeme(name, symbol, totalSupply, {value: minNativeTokenValue.sub(1)})
    ).to.be.revertedWith("Minimum native token value is required to summon");
    await expect(
        memeBase.summonThisMeme(name, symbol, MIN_TOTAL_SUPPLY.sub(1), {value: minNativeTokenValue})
    ).to.be.revertedWith("Minimum total supply is not met");
    await expect(
        memeBase.summonThisMeme(name, symbol, uint128MaxPlusOne, {value: minNativeTokenValue})
    ).to.be.revertedWith("Maximum total supply overflow");

    // Summon a new meme token - positive cases
    await expect(memeBase.summonThisMeme(name, symbol, totalSupply, {value: smallDeposit}))
        .to.emit(memeBase, "Summoned")
        .withArgs(signers[0].address, nonce1, smallDeposit)
        .and.to.emit(memeBase, "Hearted")
        .withArgs(signers[0].address, nonce1, smallDeposit);
    let totalDeposit = smallDeposit;
    let memeSummon = await memeBase.memeSummons(nonce1);
    expect(memeSummon.name).to.equal(name);
    expect(memeSummon.symbol).to.equal(symbol);
    expect(memeSummon.totalSupply).to.equal(totalSupply);
    expect(memeSummon.nativeTokenContributed).to.equal(smallDeposit);
    const memeHearterValue = await memeBase.memeHearters(nonce1, signers[0].address);
    expect(memeHearterValue).to.equal(smallDeposit);
    let accountActivity = await memeBase.mapAccountActivities(signers[0].address);
    expect(accountActivity).to.equal(1);

    // Heart a new token by other accounts - negative case
    await expect(
        memeBase.connect(signers[1]).heartThisMeme(nonce1, {value: 0})
    ).to.be.revertedWith("Native token amount must be greater than zero");
    await expect(
        memeBase.connect(signers[1]).heartThisMeme(nonce2, {value: smallDeposit})
    ).to.be.revertedWith("Meme not yet summoned");
    // Check that launch campaign meme cannot be hearted (as a pseudo test)
    await expect(
        memeBase.connect(signers[1]).heartThisMeme(nonce0, {value: smallDeposit})
    ).to.be.revertedWith("Meme already unleashed");

    // Heart a new token by other accounts - positive cases
    await expect(memeBase.connect(signers[1]).heartThisMeme(nonce1, {value: smallDeposit}))
        .to.emit(memeBase, "Hearted")
        .withArgs(signers[1].address, nonce1, smallDeposit);
    totalDeposit = totalDeposit.add(smallDeposit)
    memeSummon = await memeBase.memeSummons(nonce1);
    expect(memeSummon.nativeTokenContributed).to.equal(ethers.BigNumber.from(smallDeposit).mul(2));
    accountActivity = await memeBase.mapAccountActivities(signers[1].address);
    expect(accountActivity).to.equal(1);
    await expect(memeBase.connect(signers[2]).heartThisMeme(nonce1, {value: smallDeposit}))
        .to.emit(memeBase, "Hearted")
        .withArgs(signers[2].address, nonce1, smallDeposit);
    totalDeposit = totalDeposit.add(smallDeposit)
    memeSummon = await memeBase.memeSummons(nonce1);
    expect(memeSummon.nativeTokenContributed).to.equal(ethers.BigNumber.from(smallDeposit).mul(3));
    accountActivity = await memeBase.mapAccountActivities(signers[2].address);
    expect(accountActivity).to.equal(1);

    await expect(
        memeBase.unleashThisMeme(nonce1)
    ).to.be.revertedWith("Cannot unleash yet");

    // Global native token counters
    let totalNativeTokens = await memeBase.totalNativeTokens();
    expect(totalNativeTokens).to.equal(0);
    let totalPooledNativeTokens = await memeBase.totalPooledNativeTokens();
    expect(totalPooledNativeTokens).to.equal(0);
    let totalAscendedNativeTokens = await memeBase.totalAscendedNativeTokens();
    expect(totalAscendedNativeTokens).to.equal(0);

    // Increase time to for 24 hours+
    await helpers.time.increase(oneDay + 10);

    // Unleash the meme token - positive and negative cases
    await expect(
        memeBase.unleashThisMeme(nonce2)
    ).to.be.revertedWith("Meme not summoned");
    await memeBase.unleashThisMeme(nonce1);
//    await expect(await memeBase.unleashThisMeme(nonce1))
//        .to.emit(memeBase, "Unleashed")
//        // .withArgs(signers[0].address, null, null, null, 0)
//        .and.to.emit(memeBase, "Collected");
//        // .withArgs(signers[0].address, null, null);
    await expect(
        memeBase.unleashThisMeme(nonce1)
    ).to.be.revertedWith("Meme already unleashed");

    totalNativeTokens = await memeBase.totalNativeTokens();
    expect(totalNativeTokens).to.equal(totalDeposit);
    totalPooledNativeTokens = await memeBase.totalPooledNativeTokens();
    expect(totalPooledNativeTokens).to.lte((totalDeposit.mul(9)).div(10));
    // Still zero as all the burn funds went to the campaign
    totalAscendedNativeTokens = await memeBase.totalAscendedNativeTokens();
    expect(totalAscendedNativeTokens).to.equal(0);

    accountActivity = await memeBase.mapAccountActivities(signers[0].address);
    expect(accountActivity).to.equal(2);

    // Get first token address
    const memeToken = await memeBase.memeTokens(0);
    console.log("First new meme token contract:", memeToken);

    expect(memeSummon.nativeTokenContributed).to.equal(ethers.BigNumber.from(smallDeposit).mul(3));

    // Schedule for ascendance is zero because all the funds went to the campaign launch
    let scheduledForAscendance = await memeBase.scheduledForAscendance();
    expect(scheduledForAscendance).to.equal(0);

    let launchCampaignBalance = await memeBase.launchCampaignBalance();
    expect(launchCampaignBalance).to.equal(ethers.BigNumber.from(smallDeposit).mul(3).div(10));

    // Increase time to for 24 hours+
    await expect(
        memeBase.purgeThisMeme(memeToken)
    ).to.be.revertedWith("Purge only allowed from 24 hours after unleash");
    await helpers.time.increase(oneDay + 10);

    // Purge remaining allocation - positive and negative case
    await memeBase.purgeThisMeme(memeToken);

    //// Second test unleashing of a meme that does trigger MAGA

    // TOFIX; either get here and it passes, or we fail at Unleash above
    console.log(">>>>>---------here");

    // Summon a new meme token
    await memeBase.summonThisMeme(name, symbol, totalSupply, {value: defaultDeposit});

    // Heart a new token by other accounts
    await memeBase.connect(signers[1]).heartThisMeme(nonce2, {value: defaultDeposit});
    await memeBase.connect(signers[2]).heartThisMeme(nonce2, {value: defaultDeposit});

    // Update total deposit
    totalDeposit = totalDeposit.add(defaultDeposit.mul(3));

    // Increase time to for 24 hours+
    await helpers.time.increase(oneDay + 10);

    // Unleash the meme token
    await memeBase.unleashThisMeme(nonce2);

    const LIQUIDITY_AGNT = await memeBase.LIQUIDITY_AGNT();
    launchCampaignBalance = await memeBase.launchCampaignBalance();
    scheduledForAscendance = await memeBase.scheduledForAscendance();
    const expectedScheduledForAscendance = ethers.BigNumber.from(smallDeposit).mul(3).div(10)
        .add(ethers.BigNumber.from(defaultDeposit).mul(3).div(10))
        .sub(ethers.BigNumber.from(launchCampaignBalance));
    // this fails ever so often
    expect(scheduledForAscendance).to.equal(expectedScheduledForAscendance);
    expect(launchCampaignBalance).to.equal(LIQUIDITY_AGNT);

    totalNativeTokens = await memeBase.totalNativeTokens();
    expect(totalNativeTokens).to.equal(totalDeposit);
    totalPooledNativeTokens = await memeBase.totalPooledNativeTokens();
    expect(totalPooledNativeTokens).to.lte((totalDeposit.mul(9)).div(10));
    // Still zero as all the burn funds went to the campaign
    totalAscendedNativeTokens = await memeBase.totalAscendedNativeTokens();
    expect(totalAscendedNativeTokens).to.equal((totalDeposit.div(10)).sub(launchCampaignBalance));

    // Get campaign token
    const campaignToken = await memeBase.memeTokens(1);
    console.log("Campaign meme contract:", campaignToken);
    // Get meme token
    const memeTokenTwo = await memeBase.memeTokens(2);
    console.log("Meme token two contract:", memeToken);

    // Deployer has already collected
    await expect(
        memeBase.collectThisMeme(memeTokenTwo)
    ).to.be.reverted;

    // Collect by the first signer
    await memeBase.connect(signers[1]).collectThisMeme(memeTokenTwo);

    // Wait for 24 more hours
    await helpers.time.increase(oneDay + 10);

    // Second signer cannot collect
    await expect(
        memeBase.connect(signers[2]).collectThisMeme(memeTokenTwo)
    ).to.be.reverted;

    // Purge remaining allocation
    await memeBase.purgeThisMeme(memeTokenTwo);

    // Wait for 10 more seconds
    await helpers.time.increase(10);

    // Swap to OLAS
    const olasAmount = await memeBase.scheduledForAscendance();
    // First 127.5 ETH are collected towards campaign
    if (olasAmount.gt(0)) {
        await memeBase.scheduleForAscendance();
    }

    // Collect fees
    await memeBase.collectFees([campaignToken, memeToken, memeTokenTwo]);

    // Check the contract balance
    const baseBalance = await ethers.provider.getBalance(memeBase.address);
    expect(baseBalance).to.equal(0);
};

main()
    .then(() => process.exit(0))
    .catch(error => {
        console.error(error);
        process.exit(1);
    });
