send_telegram_message <- function(text, chat_id, bot_token){
  require(telegram)
  bot <- TGBot$new(token = bot_token)
  bot$sendMessage(text = text, chat_id = chat_id)
}

send_telegram_photo <- function(photo, caption, chat_id, bot_token){
  require(telegram)
  bot <- TGBot$new(token = bot_token)
  bot$sendPhoto(photo = photo, caption = caption, chat_id = chat_id)
}

send_telegram_document <- function(document, chat_id, bot_token){
  require(telegram)
  bot <- TGBot$new(token = bot_token)
  bot$sendDocument(document = document, chat_id = chat_id)
}

# send_telegram_message(text = "Your script is finished", 
#                       chat_id = "441084295", 
#                       bot_token = "880903665:AAE_f0i_bQRXBXJ4IR5TEuTt5C05vvaTJ5w")

# send_telegram_photo(photo = "Kitty.png", 
#                     caption = "Look at this awesome kitty!", 
#                     chat_id = "441084295", 
#                     bot_token = "880903665:AAE_f0i_bQRXBXJ4IR5TEuTt5C05vvaTJ5w")

# send_telegram_document(document = "Hardlopen.xlsx",
#                        chat_id = "441084295",
#                        bot_token = "880903665:AAE_f0i_bQRXBXJ4IR5TEuTt5C05vvaTJ5w")
