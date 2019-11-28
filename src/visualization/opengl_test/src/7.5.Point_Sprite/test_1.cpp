#include <ROS_ICLU3_v0.hpp>

// Debug
#include <iostream>
// test
#include <FILTER_LIB.h>


using TIME_STAMP::Time;
using TIME_STAMP::Period;

int main(int argc, char *argv[])
{
    Time t_start(TIME_PARAM::NOW);
    t_start.show();

    Period period_1;


    Time t_a(3.0);
    t_a.show();
    Time t_b(1.0);
    t_b.show();

    Time t_c = t_b - t_a;
    t_c.show_sec();


    std::shared_ptr<int> a_ptr;
    a_ptr.reset(new int(5));
    std::cout << "a.use_count() = " << a_ptr.use_count() << "\n";

    boost::any any_a = a_ptr;
    a_ptr.reset(new int(0));
    std::cout << "a.use_count() = " << a_ptr.use_count() << "\n";
    std::cout << "any_a.use_count = " << ( boost::any_cast< std::shared_ptr<int> >(any_a) ).use_count() << "\n";
    a_ptr = boost::any_cast< std::shared_ptr<int> >(any_a);
    std::cout << "a.use_count() = " << a_ptr.use_count() << "\n";


    std::shared_ptr<int> *a_ptr_ptr = boost::any_cast< std::shared_ptr<int> >(&any_a);
    std::cout << "a.use_count() = " << a_ptr.use_count() << "\n";

    //
    std::cout << "any_a = " << *( boost::any_cast< std::shared_ptr<int> >(any_a) ) << "\n";
    std::cout << "a.use_count() = " << a_ptr.use_count() << "\n";
    std::cout << "*a_ptr_ptr = " << *( *a_ptr_ptr ) << "\n";
    std::cout << "a.use_count() = " << a_ptr.use_count() << "\n";


    boost::any any_b = any_a;
    std::cout << "b.use_count() = " << ( boost::any_cast< std::shared_ptr<int> >(any_b) ).use_count() << "\n";
    std::cout << "a.use_count() = " << a_ptr.use_count() << "\n";

    //
    a_ptr.reset(new int(10));
    // std::cout << "b.use_count() = " << ( boost::any_cast< std::shared_ptr<int> >(any_b) ).use_count() << "\n";
    boost::any any_a_ptr = &a_ptr;
    std::cout << "a.use_count() = " << a_ptr.use_count() << "\n";
    // std::cout << "any_a = " << *( boost::any_cast< std::shared_ptr<int> >(any_a_ptr) ) << "\n";
    // std::cout << "a.use_count() = " << a_ptr.use_count() << "\n";


    Time t_end(TIME_PARAM::NOW);
    t_end.show();

    Time t_delta = t_end - t_start;
    t_delta.show_sec();

    period_1.stamp();
    period_1.show_sec();

    return 0;
}
