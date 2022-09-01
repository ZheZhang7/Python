import selenium.webdriver as webdriver;
from selenium.webdriver.support.ui import WebDriverWait;
from selenium.webdriver.support import expected_conditions as EC;
from selenium.webdriver.common.by import By;
from selenium.webdriver.support.ui import Select;
from selenium.common.exceptions import NoSuchElementException,ElementNotVisibleException,ElementNotInteractableException
import csv;

# driver放在外面是因为不会因为类的销毁而销毁，可以正常使用
driver = webdriver.Chrome(executable_path = r'D:\chromedrive\chromedriver.exe');

class TrainSpider(object):
    # 登录的url
    login_url = 'https://kyfw.12306.cn/otn/resources/login.html';
    # 个人中心的url
    personal_url = 'https://kyfw.12306.cn/otn/view/index.html';
    # 车次余票查询url
    left_ticket_url = 'https://kyfw.12306.cn/otn/leftTicket/init?linktypeid=dc';
    # 确认乘客和车次信息url
    confirm_passengers_url = 'https://kyfw.12306.cn/otn/confirmPassenger/initDc';
    def __init__(self, from_station, to_station, train_date, trains, passengers):
        '''
        :param from_station: 出发站
        :param to_station: 目的地
        :param train_date: 出发日期
        :param trains: 需要购买的车次，字典形式
        :param passengers: 购票人的姓名，需要的是一个列表
        '''
        self.from_station = from_station;
        self.to_station = to_station;
        self.train_date = train_date;
        self.trains = trains;
        self.passengers = passengers;
        self.selected_number = None;
        self.selected_seat = None;
        self.station_codes = {};
        # 初始化站点
        self.init_station_code()

    # 从station中读取name和code
    def init_station_code(self):
        with open("stations.csv", 'r', encoding='utf-8') as fp:
            reader = csv.DictReader(fp);
            for line in reader:
                name = line['name'];
                code = line['code'];
                self.station_codes[name] = code;

    # 登录
    def login(self):
        driver.get(self.login_url);
        # 等待url改变，判断是否登录成功
        WebDriverWait(driver, 60).until(
            # EC.url_contains(self.personal_url)
            EC.url_to_be(self.personal_url)
        )
        print('login successful!');

    # 车次余票查询
    def search_left_ticket(self):
        driver.get(self.left_ticket_url);
        # 关闭多余的弹窗
        x_btn = driver.find_element_by_id('gb_closeDefaultWarningWindowDialog_id');
        x_btn.click();
        # 起始站code设置
        ##找到输入框
        from_station_input = driver.find_element_by_id('fromStation');
        ##获取站点的code值
        from_station_code = self.station_codes[self.from_station];
        driver.execute_script('arguments[0].value="%s"'%from_station_code, from_station_input);
        #终点站code设置
        to_station_input = driver.find_element_by_id('toStation');
        to_station_code = self.station_codes[self.to_station];
        driver.execute_script('arguments[0].value="%s"'%to_station_code, to_station_input);
        # 设置时间
        train_date_input = driver.find_element_by_id('train_date');
        driver.execute_script('arguments[0].value="%s"' % self.train_date, train_date_input);
        #执行查询操作
        search_btn = driver.find_element_by_id('query_ticket');
        search_btn.click();
        #解析车次信息
        WebDriverWait(driver,100).until(
            EC.presence_of_element_located((By.XPATH, "//tbody[@id='queryLeftTable']/tr"))
        )
        train_trs = driver.find_elements_by_xpath("//tbody[@id='queryLeftTable']/tr[not(@datatran)]");
        is_searched = False;
        while True:
            for train_tr in train_trs:
                infos = train_tr.text.replace('\n', ' ').split(' ');
                number = infos[0];
                if number in self.trains:
                    seat_types = self.trains[number];
                    for seat_type in seat_types:
                        # 二等座
                        if seat_type == "O" and infos[1] == '复':
                            count = infos[10];
                            if count.isdigit() or count == '有':
                                is_searched = True;
                                break;
                        elif seat_type == 'O' and infos[1] != '复':
                            count = infos[9];
                            if count.isdigit() or count == '有':
                                is_searched = True;
                                break;
                        # 一等座
                        if seat_type == "M" and infos[1] == '复':
                            count = infos[9];
                            if count.isdigit() or count == '有':
                                is_searched = True;
                                break;
                        elif seat_type == 'M' and infos[1] != '复':
                            count = infos[8];
                            if count.isdigit() or count == '有':
                                is_searched = True;
                                break;
                    if is_searched:
                        self.selected_number = number;
                        order_btn = train_tr.find_element_by_xpath('.//a[@class="btn72"]');
                        order_btn.click();
                        return;
        print('query successful!')

    # 确认乘客和车次信息
    def confirm_passengers(self):
        WebDriverWait(driver,100).until(
            EC.url_to_be(self.confirm_passengers_url)
        )
        print('jump  successful!')
        # 等待乘客标签加载
        WebDriverWait(driver, 100).until(
            EC.presence_of_element_located((By.XPATH, "//ul[@id='normal_passenger_id']/li/label"))
        )
        # 确认购买车票的乘客
        passenger_labels = driver.find_elements_by_xpath("//ul[@id='normal_passenger_id']/li/label");
        for passenger_label in passenger_labels:
            name = passenger_label.text;
            if name in self.passengers:
                passenger_label.click();
        # 确认车次
        seat_select = Select(driver.find_element_by_id('seatType_1'));
        seat_types = self.trains[self.selected_number];
        for seat_type in seat_types:
            try:
                self.selected_seat = seat_type;
                seat_select.select_by_value(seat_type);
            except NoSuchElementException:
                continue;
            else:
                break;
        submit_btn = driver.find_element_by_id('submitOrder_id');
        submit_btn.click();
        #等待购买页面和确认按钮
        WebDriverWait(driver, 100).until(
            EC.presence_of_element_located((By.CLASS_NAME, 'dhtmlx_window_active'))
        )
        WebDriverWait(driver, 100).until(
            EC.element_to_be_clickable((By.ID, 'qr_submit_id'))
        )
        confirm_btn = driver.find_element_by_id('qr_submit_id');
        while confirm_btn:
            try:
                confirm_btn.click()
                confirm_btn = driver.find_element_by_id('qr_submit_id');
            except ElementNotVisibleException:
                break;
        # print("恭喜！成功抢到【%s】次列车【%s】席位，请在30分钟内完成付款！" % (self.selected_number, self.selected_seat))

    def run(self):
        # 1、登录
        self.login();
        #2、车次余票查询
        self.search_left_ticket();
        #3、确认乘客和车次信息
        self.confirm_passengers();

def main():
    # 9:商务座 M：一等座  O：二等座  3：硬卧 4：软卧 1：硬座
    spider = TrainSpider('北京', '长沙', '2021-10-25', {"G485":["O", "M"]},['张喆']);
    spider.run();

if __name__ == '__main__':
    main();

