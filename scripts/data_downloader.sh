#!/bin/bash
mkdir data_uncompressed
cd data_uncompressed

# download from drive

# train chunk 0
echo 'chunk 0'
wget --header="Host: doc-0o-64-docs.googleusercontent.com" --header="User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/66.0.3359.139 Safari/537.36" --header="Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8" --header="Accept-Language: en-GB,en;q=0.9,en-US;q=0.8,da;q=0.7" --header="Cookie: AUTH_v2fqtu1j93m3qmavsbg96vuovguel918=09282200773407452008|1526205600000|kp0qbughlqhrig61b8hk9dal1dmrq4d1" --header="Connection: keep-alive" "https://doc-0o-64-docs.googleusercontent.com/docs/securesc/4riudum6lki1td2er4in7gqvvj1t5rrp/7og6pvqqila51g7960s3n2ihdjkp5pb9/1526212800000/13594981441913648853/09282200773407452008/1lg1MCSBfifI5Hxao2OGPD7rUpi5N6V3B?e=download" -O "train_chunk0.zip" -c

# train chunk 1
echo 'chunk 1'
wget --header="Host: doc-0c-64-docs.googleusercontent.com" --header="User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/66.0.3359.139 Safari/537.36" --header="Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8" --header="Accept-Language: en-GB,en;q=0.9,en-US;q=0.8,da;q=0.7" --header="Cookie: AUTH_v2fqtu1j93m3qmavsbg96vuovguel918=09282200773407452008|1526205600000|kp0qbughlqhrig61b8hk9dal1dmrq4d1" --header="Connection: keep-alive" "https://doc-0c-64-docs.googleusercontent.com/docs/securesc/4riudum6lki1td2er4in7gqvvj1t5rrp/o0a2e5hpl7pa2kmitt56fb25lr15duub/1526212800000/13594981441913648853/09282200773407452008/1Cwu9YC3yDQiiIH6fQVO_QMqaxEjF7X8v?e=download" -O "train_chunk1.zip" -c

# train chunk 2
echo 'chunk 2'
wget --header="Host: doc-0k-64-docs.googleusercontent.com" --header="User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/66.0.3359.139 Safari/537.36" --header="Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8" --header="Accept-Language: en-GB,en;q=0.9,en-US;q=0.8,da;q=0.7" --header="Cookie: AUTH_v2fqtu1j93m3qmavsbg96vuovguel918=09282200773407452008|1526205600000|kp0qbughlqhrig61b8hk9dal1dmrq4d1" --header="Connection: keep-alive" "https://doc-0k-64-docs.googleusercontent.com/docs/securesc/4riudum6lki1td2er4in7gqvvj1t5rrp/n3pth764pp9d66ja20of0cp73qkl4o82/1526212800000/13594981441913648853/09282200773407452008/1zWRTjaLaqXdpsmgzRe_vBbwmWaG8Juzo?e=download" -O "train_chunk2.zip" -c

# train chunk 3
echo 'chunk 3'
wget --header="Host: doc-0s-64-docs.googleusercontent.com" --header="User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/66.0.3359.139 Safari/537.36" --header="Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8" --header="Accept-Language: en-GB,en;q=0.9,en-US;q=0.8,da;q=0.7" --header="Cookie: AUTH_v2fqtu1j93m3qmavsbg96vuovguel918=09282200773407452008|1526205600000|kp0qbughlqhrig61b8hk9dal1dmrq4d1" --header="Connection: keep-alive" "https://doc-0s-64-docs.googleusercontent.com/docs/securesc/4riudum6lki1td2er4in7gqvvj1t5rrp/gtkp00olc5gicrnfb932904ae1j5o50j/1526212800000/13594981441913648853/09282200773407452008/16l-fIUhI2sgNfonRYBUe1a1wfG0VoLLM?e=download" -O "train_chunk3.zip" -c

# train chunk 4 
echo 'chunk 4'
wget --header="Host: doc-04-64-docs.googleusercontent.com" --header="User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/66.0.3359.139 Safari/537.36" --header="Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8" --header="Accept-Language: en-GB,en;q=0.9,en-US;q=0.8,da;q=0.7" --header="Cookie: AUTH_v2fqtu1j93m3qmavsbg96vuovguel918=09282200773407452008|1526205600000|kp0qbughlqhrig61b8hk9dal1dmrq4d1" --header="Connection: keep-alive" "https://doc-04-64-docs.googleusercontent.com/docs/securesc/4riudum6lki1td2er4in7gqvvj1t5rrp/la4qg3amrabsoqut4n32afvkjh1nh2mc/1526212800000/13594981441913648853/09282200773407452008/1JPrz8DHc6lLZyv25xi7tRmvcK2PtGFeb?e=download" -O "train_chunk4.zip" -c

# train chunk 5
echo 'chunk 5'
wget --header="Host: doc-00-64-docs.googleusercontent.com" --header="User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/66.0.3359.139 Safari/537.36" --header="Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8" --header="Accept-Language: en-GB,en;q=0.9,en-US;q=0.8,da;q=0.7" --header="Cookie: AUTH_v2fqtu1j93m3qmavsbg96vuovguel918=09282200773407452008|1526205600000|kp0qbughlqhrig61b8hk9dal1dmrq4d1" --header="Connection: keep-alive" "https://doc-00-64-docs.googleusercontent.com/docs/securesc/4riudum6lki1td2er4in7gqvvj1t5rrp/6bad3c3hai4n3u4rh7aic70g3371pp4l/1526212800000/13594981441913648853/09282200773407452008/1HRBfchMJXY19AVDu1sOgs5aR-CAEYMkr?e=download" -O "train_chunk5.zip" -c

# train chunk 6
echo 'chunk 6'
wget --header="Host: doc-14-64-docs.googleusercontent.com" --header="User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/66.0.3359.139 Safari/537.36" --header="Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8" --header="Accept-Language: en-GB,en;q=0.9,en-US;q=0.8,da;q=0.7" --header="Cookie: AUTH_v2fqtu1j93m3qmavsbg96vuovguel918=09282200773407452008|1526205600000|kp0qbughlqhrig61b8hk9dal1dmrq4d1" --header="Connection: keep-alive" "https://doc-14-64-docs.googleusercontent.com/docs/securesc/4riudum6lki1td2er4in7gqvvj1t5rrp/mcdubj6igciqlap2m27nthar03fan5ks/1526212800000/13594981441913648853/09282200773407452008/1k5vA7-_ufmAnXYhPVqZXbROjmEnVMPDc?e=download" -O "train_chunk6.zip" -c

# validation
echo 'valid'
wget --header="Host: doc-00-64-docs.googleusercontent.com" --header="User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/66.0.3359.139 Safari/537.36" --header="Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8" --header="Accept-Language: en-GB,en;q=0.9,en-US;q=0.8,da;q=0.7" --header="Cookie: AUTH_v2fqtu1j93m3qmavsbg96vuovguel918=09282200773407452008|1526205600000|kp0qbughlqhrig61b8hk9dal1dmrq4d1" --header="Connection: keep-alive" "https://doc-00-64-docs.googleusercontent.com/docs/securesc/4riudum6lki1td2er4in7gqvvj1t5rrp/1f7n48npvvs261o5oeisr2i1ahab2jle/1526212800000/13594981441913648853/09282200773407452008/1P45QJN_Yd2PKFCE_9922jdvVVDd-XIyQ?e=download" -O "valid.zip" -c

# test
echo 'test'
wget --header="Host: doc-0k-64-docs.googleusercontent.com" --header="User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/66.0.3359.139 Safari/537.36" --header="Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8" --header="Accept-Language: en-GB,en;q=0.9,en-US;q=0.8,da;q=0.7" --header="Cookie: AUTH_v2fqtu1j93m3qmavsbg96vuovguel918=09282200773407452008|1526205600000|kp0qbughlqhrig61b8hk9dal1dmrq4d1" --header="Connection: keep-alive" "https://doc-0k-64-docs.googleusercontent.com/docs/securesc/4riudum6lki1td2er4in7gqvvj1t5rrp/sbb48fqf8jsicqhh1b69h0uqr5tmmv4q/1526212800000/13594981441913648853/09282200773407452008/1erAf4O6vTxuXesgA4vDVvtt14np6l9ws?e=download" -O "test.zip" -c

echo 'ZIP DOWNLOADING DONE'
