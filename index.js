/**
 * Simple image classifier with pre-trained model from tensorflow.js
 * 
 * Author: David Joos - drdavejoos@gmail.com
 * 
 */
require('dotenv').config();

require('@tensorflow/tfjs-backend-cpu');

const tf = require('@tensorflow/tfjs-node');

const cocoSsd = require('@tensorflow-models/coco-ssd');
const { Telegraf } = require('telegraf');
const axios = require('axios');
const util = require('util');
const fs = require('fs');

const bot = new Telegraf(process.env.BOT_TOKEN);

bot.on('text', (ctx) => {
  return ctx.reply(`Hi ${ctx.message.from.first_name}, since this is an image classifier i expect you to send an image.`);
});

bot.on('message', async (ctx) => {

  const files = ctx.message.photo;
  const fileId = files[1].file_id;

  const url = await ctx.telegram.getFileLink(fileId);
  const response = await axios(url.href, { responseType: 'stream' });

  const streamWrite = response.data.pipe(fs.createWriteStream(`./public/images/${ctx.message.from.first_name}${ctx.message.from.last_name}.jpg`));

  streamWrite.on('finish', async () => {
    const predictions = await handleClassification(ctx);

    if (predictions.length > 0) {
      const predictionClasses = predictions.map(p => p.class);
      predictionClasses.reduce((acc, cur) => acc + ', ' + cur);
      ctx.telegram.sendMessage(ctx.message.chat.id, `I am Back! I see following things: ${predictionClasses}`);
    } else {
      ctx.telegram.sendMessage(ctx.message.chat.id, `Uff, rare case, could not identify anything.`);
    }
  });

  return ctx.reply(`Hi ${ctx.message.from.first_name}, i need some time to process the image.`);

});

const handleClassification = async ctx => {

  const readImg = util.promisify(fs.readFile);

  try {
    const img = await readImg(`./public/images/${ctx.message.from.first_name}${ctx.message.from.last_name}.jpg`);
    const adaptedImg = tf.node.decodeImage(img, 3);
    const model = await cocoSsd.load();
    const predictions = await model.detect(adaptedImg);
    return predictions;
  } catch (e) {
    throw new Error('Something went wrong: ', e);
  }

}

bot.launch()

// Enable graceful stop
process.once('SIGINT', () => bot.stop('SIGINT'))
process.once('SIGTERM', () => bot.stop('SIGTERM'))
