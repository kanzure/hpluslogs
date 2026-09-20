-- auto-url.lua  (Pandoc 2.x)
local utils = require 'pandoc.utils'

-- crude but practical URL detector
local function looks_like_url (s)
  return s:match('^https?://') or
         s:match('^ftp://')    or
         s:match('^www%.')     -- add more prefixes if you like
end

-- turn one URL string into a proper Link
local function url_to_link (url)
  local target = url
  if url:match('^www%.') then target = 'https://' .. url end
  return pandoc.Link({pandoc.Str(url)}, target)
end

-- walk all Str elements inside a block
local function autolink (el)
  -- only process plain paragraphs, table cells, etc.
  if not el.content then return nil end

  local new_content = pandoc.List{}
  for _,inline in ipairs(el.content) do
    if inline.t == 'Str' and looks_like_url(inline.text) then
      new_content:insert(url_to_link(inline.text))
    else
      new_content:insert(inline)
    end
  end
  el.content = new_content
  return el
end

return {
  {Para = autolink},
  {Plain = autolink},
}

