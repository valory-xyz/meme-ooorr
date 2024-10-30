const fs = require("fs");
const globalsFile = "globals.json";
const dataFromJSON = fs.readFileSync(globalsFile, "utf8");
const parsedData = JSON.parse(dataFromJSON);

module.exports = [
    parsedData.olasAddress,
    parsedData.cusdAddress,
    parsedData.routerAddress,
    parsedData.factoryAddress,
    parsedData.minNativeTokenValue,
    parsedData.celoAddress,
    parsedData.l2TokenBridgeAddress,
    parsedData.oracleAddress
];