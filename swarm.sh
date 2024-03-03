if [ "$1" = "start" ]; then
    pm2 delete miner0 || true
    pm2 start neurons/miner.py --name miner0 -- --device cuda:0 --uid 0 --wallet.name gradient --wallet.hotkey miner0
    pm2 delete miner1 || true
    pm2 start neurons/miner.py --name miner1 -- --device cuda:1 --uid 1 --wallet.name gradient --wallet.hotkey miner1
    pm2 delete miner2 || true
    pm2 start neurons/miner.py --name miner2 -- --device cuda:2 --uid 2 --wallet.name gradient --wallet.hotkey miner2
    pm2 delete miner3 || true
    pm2 start neurons/miner.py --name miner3 -- --device cuda:3 --uid 3 --wallet.name gradient --wallet.hotkey miner3
    pm2 delete miner4 || true
    pm2 start neurons/miner.py --name miner4 -- --device cuda:4 --uid 4 --wallet.name gradient --wallet.hotkey miner4
    pm2 delete validator1 || true
    pm2 start neurons/validator.py --name validator1 -- --device cuda:5 --wallet.name gradient --wallet.hotkey validator1
    pm2 delete validator2 || true
    pm2 start neurons/validator.py --name validator2 -- --device cuda:6 --wallet.name gradient --wallet.hotkey validator2
    pm2 delete trainer || true
    pm2 start neurons/trainer.py --name trainer -- --device cuda:7
elif [ "$1" = "stop" ]; then
    pm2 delete miner0 || true
    pm2 delete miner1 || true
    pm2 delete miner2 || true
    pm2 delete miner3 || true
    pm2 delete validator1 || true
    pm2 delete validator2 || true
    pm2 delete trainer || true
    pm2 stop miner0
    pm2 stop miner1
    pm2 stop miner2
    pm2 stop miner3
    pm2 stop miner4
    pm2 stop validator1
    pm2 stop validator2
    pm2 stop trainer
fi
